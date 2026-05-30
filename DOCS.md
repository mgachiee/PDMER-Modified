# Changes from Baseline

This document summarizes the modifications made after the baseline codebase. It is intended as a high-level guide to what changed, where the changes live, and why they were added.

## Dataset preprocessing and embedding cache

- [script/dataset.py](script/dataset.py):
	- Added chunked processing for the test-set global ImageBind embedding to avoid out-of-memory errors on long tracks. This preserves the original per-segment output shape by concatenating chunk outputs in time order.
	- Aligns global embedding segments with label length by trimming segment lists to the label-derived length.
	- Adds explicit garbage collection and `torch.cuda.empty_cache()` between chunks to keep GPU memory stable.
	- Keeps the existing behavior that allows `args.test_data_only` to skip the training split during preprocessing.
- [script/dataset.sh](script/dataset.sh):
	- Helper command for running preprocessing with CPU and a named cache output directory.

## ImageBind and PDMER embedding memory usage

- [models/PDMER.py](models/PDMER.py):
	- `PDMERModel.get_embedding` now performs waveform segmentation on CPU, batches the ImageBind forward passes, and moves outputs to CPU to reduce peak GPU usage.
	- Adds explicit cleanup after each batch (`gc.collect()` and `torch.cuda.empty_cache()`).
- [models/image_bind.py](models/image_bind.py):
	- `ImageBind.get_embedding` mirrors the CPU segmentation and batched processing to reduce memory pressure.
	- Adds the same GPU cleanup between chunked batches.

## Training resume and checkpoint workflow

- [utils/args.py](utils/args.py):
	- Adds `--resume` to load a full checkpoint and continue training.
	- Adds `--save_latest` to always write a rolling `latest.pt` checkpoint.
- [utils/train.py](utils/train.py):
	- Adds full checkpoint helpers (`save_checkpoint`, `load_checkpoint`) that persist model state, optimizer state, epoch, and best validation metrics.
	- Uses a filtered state dict to exclude ImageBind weights from checkpoints.
- [train.py](train.py):
	- Wires resume logic into the training loop (restores epoch and best metrics).
	- Saves full checkpoints every `--save_every_epoch` epochs; `save_model` still writes best-model snapshots.

### Resume usage

Start a run with rolling checkpoints:

```bash
python train.py --train_name 2026-05-30-12-00-00 --save_every_epoch 1 --save_latest
```

Resume from the latest checkpoint:

```bash
python train.py --train_name 2026-05-30-12-00-00 --resume latest --save_every_epoch 1 --save_latest
```

Resume from a specific checkpoint path:

```bash
python train.py --train_name 2026-05-30-12-00-00 --resume /path/to/logs/2026-05-30-12-00-00/checkpoints/epoch_10.pt

### Start / Stop / Resume Training

- **Checkpoints location:** By default checkpoints are written to `<log_dir>/<train_name>/checkpoints/` and include files named `epoch_{N}.pt` and, when `--save_latest` is used, `latest.pt`.
- **Start a run with rolling checkpoints:** Save every epoch (recommended for safe resumes):

```bash
python train.py --train_name 2026-05-30-12-00-00 --save_every_epoch 1 --save_latest
```

- **Stop a run:** Press `Ctrl+C` to interrupt. Note: there is no automatic SIGINT auto-save implemented, so use `--save_latest` or a small `--save_every_epoch` to ensure recent progress is persisted.
- **Resume from the latest checkpoint:** The special `--resume latest` resolves to the `latest.pt` file under the matching train folder:

```bash
python train.py --train_name 2026-05-30-12-00-00 --resume latest --save_latest
```

- **Resume from a specific checkpoint file:** Provide the full path to an `epoch_{N}.pt` checkpoint:

```bash
python train.py --train_name 2026-05-30-12-00-00 --resume /path/to/logs/2026-05-30-12-00-00/checkpoints/epoch_10.pt
```

- **What is restored on resume:** `--resume` restores the saved epoch number, the model weights (filtered to exclude ImageBind internals), and the optimizer state so training continues from the correct step and learning-rate state.
- **Important notes:**
	- Checkpoints intentionally exclude ImageBind weights to reduce checkpoint size; ensure your environment can reconstruct or re-download ImageBind weights when resuming if you changed that component.
	- If you need more frequent recovery points, use `--save_every_epoch 1` or `--save_latest`.
	- Scheduler state and RNG state are not currently saved; add them if exact reproducibility is required.

```

## Housekeeping

- [.gitignore](.gitignore):
	- Ignores runtime artifacts such as logs, checkpoints, virtual environments, and dataset output folders.
- [DOCS.md](DOCS.md):
	- This document, capturing all changes from the baseline codebase.
