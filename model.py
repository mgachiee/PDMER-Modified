import json
import torch
import torch.nn.functional as F

from pathlib import Path
from models.PDMER import PDMERModel
from utils.inference import build_batch, slide_inference

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

    audio_file_path_list = [str(file) for file in dataset_folder.iterdir() if file.is_file and file.suffix in [".wav", ".mp3", ".flac"]]

    print("Build batch started")
    embedding, _ = build_batch(
        audio_file_path_list,
        imagebind_model=None,
        device=device,
    )

    print("Preparing Output") 
    with torch.no_grad():
        # 1. Setup Keys and Limits
        AUDIO_KEY = "log_mel_spectrogram"
        NATIVE_WINDOW = 60

        # Extract the audio tensor safely to check dimensions
        if AUDIO_KEY in embedding:
            audio_tensor = embedding[AUDIO_KEY]
        else:
            audio_tensor = next(iter(embedding.values()))

        seq_len = audio_tensor.size(1)
        batch_size = audio_tensor.size(0)

        # 2. Choose the Inference Path
        if seq_len > NATIVE_WINDOW:
            batch_arousal = []
            batch_valence = []
            
            # Process each song in the batch one by one to accommodate slide_inference
            for i in range(batch_size):
                # Isolate exactly one song, maintaining a batch size of 1
                single_song_dict = {
                    key: tensor[i : i + 1] 
                    for key, tensor in embedding.items() 
                    if tensor is not None
                }
                
                # Run sliding inference on the single track
                single_inference_result = slide_inference(model, single_song_dict)
                
                # Collect both Arousal (index 0) and Valence (index 1)
                batch_arousal.append(single_inference_result["model_output"][0])
                batch_valence.append(single_inference_result["model_output"][1])
                
            # Combine individual tracks across the batch dimension -> Shape: [batch_size, seq_len]
            final_arousal = torch.cat(batch_arousal, dim=0)
            final_valence = torch.cat(batch_valence, dim=0)
            
            # FIX: Add back the trailing channel dimension to restore 3D shape -> [batch_size, seq_len, 1]
            # This completely satisfies downstream 3D indexing requirements
            final_arousal = final_arousal.unsqueeze(-1)
            final_valence = final_valence.unsqueeze(-1)
            
            # Reconstruct the exact dictionary structure expected by your annotation script
            output = {
                "model_output": [final_arousal, final_valence],
                "attention_maps": None  # slide_inference does not track batched attention maps
            }
            
        else:
            # If the sequence is short enough, pass it directly to the native model
            output = model(embedding)

    arousal = output["model_output"][0]
    valence = output["model_output"][1]

    print("Arousal:", arousal.shape)
    print("Valence:", valence.shape)
    print(arousal)
    print(valence)

if __name__ == "__main__":
    print("Annotation started...")
    train()
    print("Annoation completed... ")