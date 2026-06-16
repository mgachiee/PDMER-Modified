import json
import torch
import torch.nn.functional as F
import pandas as pd

from pathlib import Path
from models.PDMER import PDMERModel
from models.image_bind import ImageBind
from utils.inference import build_batch, slide_inference
from utils.music.util import generate_split_duration_list

torch.set_printoptions(sci_mode=False, precision=4)

def predict_with_sliding_window(model, full_embedding, window_size=267, stride=267):
    """
    Slices a long audio feature tensor into model-compatible chunks,
    gets predictions for each, and stitches them back together chronologically.

    Args:
        model: The initialized and loaded PDMER model.
        full_embedding: Tensor of shape [1, Total_Frames, 51, 128].
        window_size: The strict frame length the model was trained on (267).
        stride: How far to move the window. If stride == window_size, there is no overlap.
    """
    # Ensure we are working with a single file (Batch size = 1) to avoid complex batch-padding
    assert full_embedding.size(0) == 1, "Sliding window inference requires batch_size=1"

    # Remove batch dimension for easier slicing: [Total_Frames, 51, 128]
    features = full_embedding.squeeze(0)
    total_frames = features.size(0)
    
    all_chunk_predictions = []

    # Slide the window across the time axis
    for start_idx in range(0, total_frames, stride):
        end_idx = start_idx + window_size
        chunk = features[start_idx:end_idx]

        actual_chunk_len = chunk.size(0)

        # Pad the final chunk with zeros if it is shorter than the required window size
        if actual_chunk_len < window_size:
            pad_size = window_size - actual_chunk_len
            # F.pad adds padding from the last dimension forward: (dim2, dim1, dim0)
            chunk = F.pad(chunk, (0, 0, 0, 0, 0, pad_size))

        # Re-add batch dimension so the model accepts it: [1, window_size, 51, 128]
        chunk_batch = chunk.unsqueeze(0)

        # Pass the perfectly sized 30-second chunk to the model
        with torch.no_grad():
            predictions = model(chunk_batch)
            
            # If we padded the input, we must discard the garbage predictions 
            # generated for the zero-padded frames at the very end of the song.
            # (Assuming model output shape is [1, Time_Steps, Num_Classes])
            if actual_chunk_len < window_size:
                # Slice the time dimension to keep only valid frames
                if isinstance(predictions, tuple): 
                    # Handle cases where the model returns multiple outputs (like attention maps)
                    predictions = predictions[0][:, :actual_chunk_len, :]
                else:
                    predictions = predictions[:, :actual_chunk_len, :]

            all_chunk_predictions.append(predictions)

    # Stitch all chunk predictions together along the time dimension (dim=1)
    # The final shape will be [1, Original_Total_Frames, Num_Classes]
    final_timeline_prediction = torch.cat(all_chunk_predictions, dim=1)
    
    return final_timeline_prediction

def train():
    device = torch.device("cpu") # Change to preferred device if needed (e.g., "cuda:0" for specific GPU)

    # Adjust as needed to point to your actual training logs and model checkpoints
    train_name = "va-annotator-model"
    config_path = f"logs/{train_name}/config.json"
    model_path = f"logs/{train_name}/models/latest.pt"
    dataset_folder = Path("data/dataset") # Adjust to your actual dataset folder containing audio files

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = PDMERModel(
        device=device,
        query_embed_dim=cfg["model_query_embed_dim"],
        num_attention_heads=cfg["model_num_attention_heads"],
        num_hidden_layers=cfg["model_num_hidden_layers"],
        segmentation_duration=cfg["model_segmentation_duration"],
        feature_num_per_audio=cfg["model_feature_num_per_audio"],
        train_audio_duration=cfg["model_train_audio_duration"],
        dropout_rate=cfg["dropout_rate"],
        intermediate_size=cfg["intermediate_size"],
        hidden_act=cfg["hidden_act"],
        max_position_embeddings=cfg["max_position_embeddings"],
        embed_dim=cfg["embed_dim"],
        audio_input_key=cfg["audio_input_key"],
        local_context_length=cfg["local_context_length"],
        global_context_length=cfg["global_context_length"],
        position_embedding_type=cfg["position_embedding_type"],
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load ImageBind once to avoid reloading for every file
    print("Loading ImageBind model...")
    imagebind_model = ImageBind(device=device).to(device=device).eval()

    audio_file_path_list = [str(file) for file in dataset_folder.iterdir() if file.is_file and file.suffix in [".wav", ".mp3", ".flac"]]

    results = []
    output_path = Path("data/dataset/annotations.csv")

    for path in audio_file_path_list:
        print(f"Processing: {path}")
        
        # Calculate actual length for this single file to handle padding correctly
        durations = generate_split_duration_list(path, begin_time=0)
        actual_length = len(durations)
        
        # Build batch for ONE file using the pre-loaded imagebind_model
        embedding, _ = build_batch(
            [path],
            imagebind_model=imagebind_model,
            device=device,
        )

        with torch.no_grad():
            # Setup Keys and Limits
            AUDIO_KEY = "log_mel_spectrogram"
            NATIVE_WINDOW = 60

            # Extract the audio tensor safely to check dimensions
            audio_tensor = embedding[AUDIO_KEY] if AUDIO_KEY in embedding else next(iter(embedding.values()))
            seq_len = audio_tensor.size(1)

            # Choose the Inference Path
            if seq_len > NATIVE_WINDOW:
                # Slice away any potential padding from build_batch to ensure clean sliding window
                single_song_dict = {
                    key: (tensor[:, :actual_length, :] if key != "global_imagebind_audio_embedding" else tensor)
                    for key, tensor in embedding.items() 
                    if tensor is not None
                }
                
                # Run sliding inference on the single track
                single_inference_result = slide_inference(model, single_song_dict)
                arousal = single_inference_result["model_output"][0]
                valence = single_inference_result["model_output"][1]
                
            else:
                # If the sequence is short enough, pass it directly to the native model
                output_dict = model(embedding)
                arousal = output_dict["model_output"][0]
                valence = output_dict["model_output"][1]

        # Extract timelines (Batch size is guaranteed to be 1 here)
        # Squeeze to remove batch and trailing channel dimensions -> [seq_len]
        a_vals = arousal.squeeze().tolist()
        v_vals = valence.squeeze().tolist()
        
        # Ensure we only keep the non-padded frames if the model returned more
        if isinstance(a_vals, list):
            a_vals = a_vals[:actual_length]
            v_vals = v_vals[:actual_length]
        else:
            # Handle case where squeeze might result in a scalar for extremely short audio
            a_vals = [a_vals]
            v_vals = [v_vals]

        results.append({
            "file_path": path,
            "arousal_mean": sum(a_vals) / len(a_vals) if a_vals else 0,
            "valence_mean": sum(v_vals) / len(v_vals) if v_vals else 0,
            "arousal_timeline": a_vals,
            "valence_timeline": v_vals
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    print("Annotation started...")
    train()
    print("Annotation completed... ")