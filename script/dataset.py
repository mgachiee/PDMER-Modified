import os
import sys
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.PDMER import PDMERModel
from utils.args import parse_args
from utils.DEAM.dataset import DEAMDataset, get_label_true_shape
from utils.logger import setup_logging
from utils.music.util import generate_split_duration_list, get_audio_log_mel_spec
from utils.PMEmo.dataset import PMEmoDataset

args = parse_args(
    description="Extract audio embedding from the DEAM dataset",
    arguments=[
        {
            "args": ["--cache_output_name"],
            "kwargs": {
                "type": str,
                "default": "audio_embedding_test",
                "help": "The root directory of the dataset",
            },
        }
    ],
)
setup_logging()

train_dataset = (PMEmoDataset if args.dataset_name == "PMEmo" else DEAMDataset)(
    root=args.dataset_root, using_cache_tensor=False, is_train_data=True
)
test_dataset = (PMEmoDataset if args.dataset_name == "PMEmo" else DEAMDataset)(
    root=args.dataset_root, using_cache_tensor=False, is_train_data=False
)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_data_loader = DataLoader(test_dataset, batch_size=1)

model: PDMERModel = PDMERModel(
    device=args.device,
    query_embed_dim=args.model_query_embed_dim,
    num_attention_heads=args.model_num_attention_heads,
    num_hidden_layers=args.model_num_hidden_layers,
    segmentation_duration=args.model_segmentation_duration,
    feature_num_per_audio=args.model_feature_num_per_audio,
    train_audio_duration=args.model_train_audio_duration,
    dropout_rate=args.dropout_rate,
).to(args.device)

os.makedirs(
    os.path.join(args.dataset_root, "DEAM_audio", args.cache_output_name), exist_ok=True
)

duration = 30.0


def process_batch(
    batch,
    is_train_data: bool,
    split_name: str = "",
    batch_index: int = None,
    total_batches: int = None,
):
    samples, labels = batch
    if batch_index is not None and total_batches is not None:
        percent = (batch_index + 1) / total_batches * 100
        tqdm.write(
            f"[{split_name}] Batch {batch_index + 1}/{total_batches} "
            f"({percent:.1f}%) - annotating {len(samples)} sample(s)"
        )

    duration_list = [
        generate_split_duration_list(
            sample,
            sample_rate=44100,
            length_each_clip=0.5,
            begin_time=15.0,
            end_time=(
                15.0 + args.model_feature_num_per_audio // 2
                if is_train_data and args.dataset_name != "PMEmo"
                else None
            ),
        )
        for sample in samples
    ]

    image_bind_duration_list = [
        generate_split_duration_list(
            sample,
            sample_rate=44100,
            length_each_clip=args.model_segmentation_duration,
            begin_time=15.0,
            end_time=(
                15.0 + args.model_feature_num_per_audio // 2
                if is_train_data and args.dataset_name != "PMEmo"
                else None
            ),
            slide_start=0,
        )
        for sample in samples
    ]

    for i in range(len(duration_list)):
        assert len(duration_list[i]) == len(
            image_bind_duration_list[i]
        ), f"{samples[i]}, len(duration_list[i]): {len(duration_list[i])}, len(image_bind_duration_list[i]): {len(image_bind_duration_list[i])}"

    a_labels, v_labels = get_label_true_shape(labels[0]), get_label_true_shape(
        labels[1]
    )
    for i in range(len(a_labels)):
        assert v_labels[i] <= len(duration_list[i]) and a_labels[i] <= len(
            duration_list[i]
        ), f"{samples[i]}, a_labels[i]: {a_labels[i]}, v_labels[i]: {v_labels[i]}, len(duration_list[i]): {len(duration_list[i])}"

        if len(duration_list[i]) > a_labels[i] or len(duration_list[i]) > v_labels[i]:
            duration_list[i] = duration_list[i][: min(a_labels[i], v_labels[i])]

        assert min(a_labels[i], v_labels[i]) == len(
            duration_list[i]
        ), f"{samples[i]}, a_labels[i]: {a_labels[i]}, v_labels[i]: {v_labels[i]}, len(duration_list[i]): {len(duration_list[i])}"

    # Get the embdding of the ImageBind, NOTE: this will remove the first 15s
    imagebind_audio_embedding = model.get_embedding(
        samples,
        audio_segmentation_list=image_bind_duration_list,
        clip_audio=(
            (15, 15 + args.model_feature_num_per_audio // 2)
            if is_train_data and args.dataset_name != "PMEmo"
            else (15, None)
        ),
    )

    # MODIFIED
    # Added global imagebind audio embedding, which will be used in the model as a global feature, NOTE: this will remove the first 15s
    import gc

    if is_train_data:
        global_imagebind_audio_embedding = model.get_embedding(
            samples,
            audio_segmentation_list=[
                [(15, 15 + args.model_feature_num_per_audio // 2)]
                for _ in range(len(samples))
            ],
            clip_audio=(15, 15 + args.model_feature_num_per_audio // 2),
        )
    else:
        CHUNK_SIZE = 60  # tune if needed
        global_embeds = []
        for idx, sample in enumerate(samples):
            segs = generate_split_duration_list(
                sample,
                sample_rate=44100,
                length_each_clip=30,
                begin_time=15.0,
                end_time=None,
                slide_start=0,
            )
            segs = segs[: len(duration_list[idx])]  # keep alignment with labels
            if len(segs) == 0:
                segs = [(15, 15 + args.model_feature_num_per_audio // 2)]

            parts = []
            chunk_ranges = range(0, len(segs), CHUNK_SIZE)
            for s in tqdm(
                chunk_ranges,
                total=len(range(0, len(segs), CHUNK_SIZE)),
                desc=f"Global embedding chunks {os.path.basename(sample)}",
                leave=False,
            ):
                sub = segs[s : s + CHUNK_SIZE]
                out = model.get_embedding([sample], audio_segmentation_list=[sub], clip_audio=(15, None))
                parts.append(out.detach().cpu())  # move chunk result to CPU immediately
                del out
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            full = torch.cat(parts, dim=1)  # (1, total_segments, feat)
            global_embeds.append(full)

        global_imagebind_audio_embedding = torch.cat(global_embeds, dim=0)  # (batch, total_segments, feat)
    # ---------------------------------------------------------------------

    # Get the log mel spectrogram of the audio
    log_mel_spectrogram = get_audio_log_mel_spec(
        samples,
        frame_length=60,  # milliseconds
        frame_shift=10,  # milliseconds
        n_mels=128,
        duration=duration_list,
    )

    for i in range(len(samples)):
        torch.save(
            {
                "imagebind_audio_embedding": imagebind_audio_embedding[
                    i
                ],  # NOTE: This isn't used in the model
                # "fbank_feature": fbank_feature[i],
                "log_mel_spectrogram": log_mel_spectrogram[i],
                "global_imagebind_audio_embedding": global_imagebind_audio_embedding[i],
            },
            samples[i]
            .replace("wav_audio", args.cache_output_name)
            .replace("wav", "pt"),
        )


# MODIFIED
# Added option to only process test data, which is useful when we have already processed the train data and only want to process the test data, or when we want to use the pre-processed train data and only want to process the test data
if not args.test_data_only:
    for batch_index, batch in enumerate(
        tqdm(
            train_data_loader,
            desc="Extracting train dataset's audio embedding",
            unit="batch",
        )
    ):
        try:
            process_batch(
                batch,
                is_train_data=True,
                split_name="train",
                batch_index=batch_index,
                total_batches=len(train_data_loader),
            )
        except Exception as e:
            logging.exception(f"Error processing batch with samples {batch[0]}: {e}")

for batch_index, batch in enumerate(
    tqdm(
        test_data_loader,
        desc="Extracting test dataset's audio embedding",
        unit="batch",
    )
):
    try:
        process_batch(
            batch,
            is_train_data=False,
            split_name="test",
            batch_index=batch_index,
            total_batches=len(test_data_loader),
        )
    except Exception as e:
        logging.exception(f"Error processing batch with samples {batch[0]}: {e}")
