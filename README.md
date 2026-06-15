# Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning (DSAML)
<p align="center">
    <a href="https://github.com/Littleor/Personalized-DMER/blob/main/LICENSE" target="blank">
    <img src="https://img.shields.io/github/license/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator license" />
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/fork" target="blank">
    <img src="https://img.shields.io/github/forks/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator forks"/>
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/stargazers" target="blank">
    <img src="https://img.shields.io/github/stars/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator stars"/>
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/issues" target="blank">
    <img src="https://img.shields.io/github/issues/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator issues"/>
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/pulls" target="blank">
    <img src="https://img.shields.io/github/issues-pr/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator pull-requests"/>
    </a>
</p>

[[Project Website](https://littleor.github.io/PDMER/)] | [[Paper](https://arxiv.org/abs/2412.19200)]

![Model Architecture](./static/images/Model-Architecture.png)

Here is the core implementation of the DSAML model in the paper "Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning", which is accepted by the AAAI 25.

## Get Start

### Prerequisites

* Python >= 3.8.5, < 3.9
* PyTorch >= 2.2.1

### Installation

```bash
conda env create -f environment.yml
conda activate PDMER
pip install -r requirements.txt
```

### Dataset Download
We need to download the [DEAM](https://cvml.unige.ch/databases/DEAM/) dataset and unzip both the audio and annotation files. 
Specifically, you need create `DEAM_Annotations` and `DEAM_audio` folders in the root directory of the dataset root folder, and put the annotation and audio files in the corresponding folders. The final file structure should be like this:

```txt
DEAM
├── DEAM_Annotations
│   ├── annotations
├── DEAM_audio
│   ├── MEMD_audio
└── features (This is Optional)
    └── features
```

Then we need to preprocess the dataset, but before we do that, we need to create the `.env` file.

### Environment Variables

After downloading the dataset, you need to create a `.env` file in the root directory of the project. The `.env` file should contain the following environment variables:

```env
# The directory to save the logs
LOG_DIR="./logs"    

# The directory to save the audio embedding for DEAM dataset
AUDIO_EMBEDDING_DIR_NAME="feature_embedding"    
# The path to the DEAM dataset
DATASET_PATH="/your/path/to/DEAM"    

# The key to the audio input in the dataset, please keep this
AUDIO_INPUT_KEY="log_mel_spectrogram"
```

You should modify the `DATASET_PATH` and `PMEMO_DATASET_PATH` to the path where you store the DEAM and PMEmo dataset.

### Dataset Preprocessing

In order to speed up the training process, we need to preprocess the dataset. You can run the following command to preprocess the dataset:

```bash
./scripts/dataset.sh
# If you want to use specific GPU, you can add the following command
# CUDA_VISIBLE_DEVICES=1 ./scripts/dataset.sh
```

This process will take about one hour, depending on your machine.

### Train
After the dataset is preprocessed, you can train the model by running the following command:

```bash
# For DMER Task
python train.py --device "cuda:0" --not_using_maml
# For PDMER Task
python train.py --device "cuda:0" --using_personalized_data_train --using_personalized_data_validate
```

### Inference

#### Quick Start (Code Snippet)
For inference on specific files, you can use the following code snippet:

```python
from utils.inference import build_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and checkpoint before inference
# model = PDMERModel(device=device).to(device)
# model.load_state_dict(torch.load("path/to/checkpoint.pth"))

audio_file_path_list = [
    "/path/to/audio1.wav",
    "/path/to/audio2.wav",
]

# Build the input batch.
embedding, _ = build_batch(
    audio_file_path_list,
    imagebind_model=None,   # If there are no ImageBind instances, set it to None, and it will auto load the model
    device=device,
)

print("\n Build batch embedding:")
for key, value in embedding.items():
    print("\t", key, value.shape)

print("Result:")
output = model(embedding)
print("Arousal: ", output["model_output"][0].shape) # The first element is the arousal prediction, [batch_size, 2 * second]
print("Valence: ", output["model_output"][1].shape) # The second element is the valence prediction, [batch_size, 2 * second]
```

#### Batch & Long-Track Inference (`model.py`)
The `model.py` script provides a production-ready way to perform inference on folders of audio files. It is specifically optimized for long audio tracks and robust memory management.

**Structure & Features:**
- **Automatic Configuration**: Dynamically loads model hyperparameters from `logs/va-annotator-model/config.json`.
- **Checkpoint Loading**: Loads the latest trained weights from `logs/va-annotator-model/models/latest.pt`.
- **Smart Slicing**: Automatically detects if an audio track exceeds the model's native window (60 frames/30 seconds) and applies a sliding window inference (`slide_inference`) to prevent OOM and ensure continuity.
- **Batch Processing**: Scans the `data/dataset` folder for audio files (supporting `.wav`, `.mp3`, and `.flac`).
- **Device Support**: Defaults to CPU for compatibility, but easily configurable for GPU in the script.

**Usage:**
1. Ensure your trained model and config are in `logs/va-annotator-model/`.
2. Place your target audio files (`.wav`, `.mp3`, or `.flac`) in `data/dataset/`.
3. Run the script:
   ```bash
   python model.py
   ```
   The script will print the Arousal and Valence timeline shapes and values for each file.


## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@misc{zhang2024personalizeddynamicmusicemotion,
      title={Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning}, 
      author={Dengming Zhang and Weitao You and Ziheng Liu and Lingyun Sun and Pei Chen},
      year={2024},
      eprint={2412.19200},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2412.19200}, 
}
```

## Changes from Baseline (this fork)

This repository contains a set of targeted modifications made on top of the original baseline codebase to improve memory stability when processing long audio files, make dataset preprocessing more robust, and add a resumable training workflow. For full, executable examples and usage notes see `DOCS.md`.

- **Dataset preprocessing and embedding cache**
    - `script/dataset.py`: added chunked processing for the test-set global ImageBind embedding (avoids OOM on long tracks), alignment/trimming of segment lists to label length, explicit garbage collection and `torch.cuda.empty_cache()` between chunks, and optional `--test_data_only` behavior.
    - `script/dataset.sh`: helper wrapper for running preprocessing and producing cache outputs.

- **ImageBind and PDMER embedding memory usage**
    - `models/image_bind.py`: compute waveform segmentation on CPU, batch ImageBind forward passes, move per-chunk outputs back to CPU, and free GPU caches to reduce peak GPU memory.
    - `models/PDMER.py`: `get_embedding` updated to mirror the memory-safe, chunked embedding strategy.

- **Training resume and checkpoint workflow**
    - `utils/args.py`: added `--resume` (path or `latest`) and `--save_latest` CLI flags.
    - `utils/train.py`: added `save_checkpoint`, `load_checkpoint`, and a filtered model-state helper to exclude large ImageBind internals from checkpoints.
    - `train.py`: wired resume logic into training, periodic checkpoint saving, and a SIGINT/SIGTERM handler that saves a rolling `latest.pt` checkpoint on Ctrl+C.

- **Docs & notes**
    - `DOCS.md`: contains a detailed changelog, start/stop/resume examples, and recommendations for reprocessing long tracks safely.

## Attribution and forking

This repository is a fork that modifies a prior baseline project and uses pretrained components which were originally developed by other authors. When you publish this fork or create a new repository for it, please retain clear attribution to the original project and to the creators of any pretrained models used (for example, the ImageBind authors if you use that model). Add the original project's repository link and citation in the new repo's description and LICENSE file as appropriate.
