import logging
import os
import random
import signal
import sys

import learn2learn as l2l
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from models.PDMER import PDMERModel
from utils.args import parse_args
from utils.dataset import (
    generate_personalized_task_from_dataset,
    generate_task_from_dataset,
)
from utils.DEAM.dataset import DEAMDataset
from utils.logger import setup_logging
from utils.PMEmo.dataset import PMEmoDataset
from utils.train import (
    fast_adapter,
    get_average_score,
    get_dataloader_with_split_dataset,
    get_loss_with_batch,
    get_optimizer,
    print_params_info,
    save_config,
    save_checkpoint,
    save_model,
    load_checkpoint,
    validate_model,
)


def train():
    setup_logging()
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    support_dataset_dict = {
        "PMEmo": PMEmoDataset,
        "DEAM": DEAMDataset,
    }

    if args.dataset_name not in support_dataset_dict.keys():
        raise ValueError(
            f"Dataset name {args.dataset_name} is not supported, only support {support_dataset_dict.keys()}"
        )

    train_dataset = support_dataset_dict[args.dataset_name](
        root=args.dataset_root,
        device=args.device,
        audio_embedding_dir_name=args.audio_embedding_dir_name,
        loading_cache_tensor_in_device=args.loading_cache_tensor_in_device,
        is_train_data=True,
        loading_data_before_use=args.loading_data_before_use,
        using_cache_tensor=not args.not_using_cache_tensor,
        using_personalized_data=args.using_personalized_data_train,
    )
    test_dataset = support_dataset_dict[args.dataset_name](
        root=args.dataset_root,
        device=args.device,
        audio_embedding_dir_name=args.audio_embedding_dir_name,
        loading_cache_tensor_in_device=args.loading_cache_tensor_in_device,
        is_train_data=False,
        using_cache_tensor=not args.not_using_cache_tensor,
        loading_data_before_use=args.loading_data_before_use,
        using_personalized_data=args.using_personalized_data_validate,
    )

    train_loader, test_loader = get_dataloader_with_split_dataset(
        train_dataset, test_dataset, args.batch_size
    )

    model: nn.Module = PDMERModel(
        device=args.device,
        query_embed_dim=args.model_query_embed_dim,
        num_attention_heads=args.model_num_attention_heads,
        num_hidden_layers=args.model_num_hidden_layers,
        segmentation_duration=args.model_segmentation_duration,
        feature_num_per_audio=args.model_feature_num_per_audio,
        train_audio_duration=args.model_train_audio_duration,
        dropout_rate=args.dropout_rate,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.max_position_embeddings,
        embed_dim=args.embed_dim,
        audio_input_key=args.audio_input_key,
        local_context_length=args.local_context_length,
        global_context_length=args.global_context_length,
        position_embedding_type=args.position_embedding_type,
    ).to(args.device)

    maml = (
        l2l.algorithms.MAML(model, lr=args.meta_lr, first_order=True)
        if not args.not_using_maml
        else None
    )
    logging.info(
        f"We will train the model with {'MAML' if maml is not None and not args.not_using_maml else 'Regular'} train mode."
    )

    print_params_info(model)

    criterion = nn.SmoothL1Loss()

    # We use `args.optimizer` to specify the optimizer
    optimizer: torch.optim.Optimizer = get_optimizer(
        args.optimizer,
        (maml if maml is not None else model).parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    logging.info(
        f"Using {args.optimizer}, lr: {args.lr}, weight_decay: {args.weight_decay}, dropout rate: {args.dropout_rate}"
    )

    # Create a SummaryWriter for logging
    writer: SummaryWriter = SummaryWriter(
        log_dir=os.path.join(args.log_dir, args.train_name)
    )
    save_config(args=args, writer=writer)

    valid_every_epoch = 10 if maml is not None else 1

    min_val_loss = float("inf")
    max_average_score = 0.0
    train_loss = 0.0
    start_epoch = 0

    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            resume_path = os.path.join(
                args.log_dir, args.train_name, "checkpoints", "latest.pt"
            )
        start_epoch, best = load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=resume_path,
            device=args.device,
            strict=True,
        )
        min_val_loss = best.get("min_val_loss", min_val_loss)
        max_average_score = best.get("max_average_score", max_average_score)
        logging.info(
            f"Resumed from {resume_path} at epoch {start_epoch}, "
            f"min_val_loss={min_val_loss}, max_average_score={max_average_score}"
        )

    # Mutable references used by the SIGINT handler to know current state
    epoch_ref = {"current": start_epoch}
    best_ref = {
        "min_val_loss": min_val_loss,
        "max_average_score": max_average_score,
    }

    class CheckpointOnSignal:
        def __init__(self, model, optimizer, args, epoch_ref, best_ref):
            self.model = model
            self.optimizer = optimizer
            self.args = args
            self.epoch_ref = epoch_ref
            self.best_ref = best_ref

        def handler(self, signum, frame):
            logging.info(f"Signal {signum} received — saving checkpoint before exit.")
            try:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=self.epoch_ref["current"],
                    train_name=self.args.train_name,
                    log_dir=self.args.log_dir,
                    best={
                        "min_val_loss": self.best_ref.get("min_val_loss"),
                        "max_average_score": self.best_ref.get("max_average_score"),
                    },
                    save_latest=True,
                )
                logging.info("Checkpoint saved successfully on signal.")
            except Exception as e:
                logging.exception(f"Failed to save checkpoint on signal: {e}")
            finally:
                sys.exit(0)

    # Register the signal handler so Ctrl+C triggers a checkpoint save
    signal_saver = CheckpointOnSignal(model=model, optimizer=optimizer, args=args, epoch_ref=epoch_ref, best_ref=best_ref)
    signal.signal(signal.SIGINT, signal_saver.handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_saver.handler)

    for epoch in trange(start_epoch, args.num_epochs):
        # update epoch_ref for the signal handler
        epoch_ref["current"] = epoch
        # Train the model
        model.train()
        if maml is not None:
            maml.train()
        if hasattr(optimizer, "train"):
            optimizer.train()

        if maml is not None:
            train_loss = maml_train(**locals())
        else:
            train_loss = regular_train(**locals())

        if epoch % valid_every_epoch == 0:
            min_val_loss, max_average_score = validate(**locals())
            # keep best_ref in sync for the signal handler
            best_ref["min_val_loss"] = min_val_loss
            best_ref["max_average_score"] = max_average_score

        if args.save_every_epoch > 0 and epoch % args.save_every_epoch == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_name=args.train_name,
                log_dir=args.log_dir,
                best={
                    "min_val_loss": min_val_loss,
                    "max_average_score": max_average_score,
                },
                save_latest=args.save_latest,
            )


def validate(
    *,
    maml,
    model,
    test_dataset,
    optimizer,
    criterion,
    writer,
    train_loss,
    epoch,
    args,
    min_val_loss,
    max_average_score,
    valid_every_epoch,
    **kwargs,
):
    # Validation the model
    if hasattr(optimizer, "eval"):
        optimizer.eval()

    val_loss, val_score = validate_model(
        model=maml if maml is not None else model,
        dataset=test_dataset,
        criterion=criterion,
        writer=writer,
        val_mode="Validation",
        epoch=epoch // valid_every_epoch,
        args=args,
        min_loss=min_val_loss,
        is_maml=not args.not_using_maml,
    )
    average_score = get_average_score(val_score)
    logging.info(
        f"Epoch {epoch // valid_every_epoch}/{args.num_epochs},  Train Loss: {round(train_loss, 3)}, Val Loss: {round(val_loss, 3)}, Average Score: {round(average_score, 3)} , Score: { {k: round(v, 3) for k, v in val_score.items()} }"
    )
    if min_val_loss >= val_loss:
        min_val_loss = val_loss

    if max_average_score <= average_score:
        max_average_score = average_score

    # Save the model
    # epoch % args.save_every_epoch == 0 or
    if max_average_score <= average_score:
        logging.info("Saving model")
        save_model(
            model, epoch // valid_every_epoch, args.train_name, log_dir=args.log_dir
        )
    return min_val_loss, max_average_score


def optimize_model(*, train_loss, step, score, writer, optimizer):
    optimizer.step()

    # Log training loss to TensorBoard
    writer.add_scalar("Loss/Training", train_loss, step)

    # Log average score
    average_score = get_average_score(score)

    writer.add_scalar(f"AverageScore/Training", average_score, step)

    # Log training score to TensorBoard
    for key, value in score.items():
        writer.add_scalar(f"{key}/Training", value, step)
    # Log lr
    writer.add_scalar("Learn Rate", optimizer.param_groups[0]["lr"], step)

    optimizer.zero_grad()


def maml_train(
    *,
    train_dataset,
    args,
    maml,
    criterion,
    writer,
    optimizer,
    epoch,
    **kwargs,
):
    train_loss = 0.0
    for task in (
        generate_personalized_task_from_dataset
        if args.using_personalized_data_train
        else generate_task_from_dataset
    )(
        dataset=train_dataset,
        num_task=args.meta_batch_count,
        batch_size=args.batch_size,
        num_shot=args.num_shot,
    ):
        # for i in range(args.meta_batch_count):  # meta_step
        # learner = maml.clone()
        support_set, query_set = task
        loss, score = fast_adapter(
            maml=maml,
            criterion=criterion,
            args=args,
            support_set=support_set,
            query_set=query_set,
            writer=writer,
        )
        loss.backward()
        train_loss += loss.item()

    train_loss /= args.meta_batch_count

    for p in maml.parameters():
        p.grad.data.mul_(1.0 / args.meta_batch_count)

    optimize_model(
        train_loss=train_loss,
        step=epoch,
        score=score,
        writer=writer,
        optimizer=optimizer,
    )

    return train_loss


def regular_train(
    *,
    train_loader,
    model,
    args,
    criterion,
    writer,
    optimizer,
    epoch,
    **kwargs,
):
    train_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        optimizer.zero_grad()

        loss, score = get_loss_with_batch(
            model=model,
            batch=batch,
            criterion=criterion,
            device=args.device,
            audio_input_key=args.audio_input_key,
            args=args,
        )[:2]
        train_loss += loss.item()
        loss.backward()

        optimize_model(
            train_loss=loss.item(),
            step=epoch * len(train_loader) + i,
            score=score,
            writer=writer,
            optimizer=optimizer,
        )
    train_loss /= len(train_loader)
    return train_loss


if __name__ == "__main__":
    load_dotenv(override=True)
    train()
