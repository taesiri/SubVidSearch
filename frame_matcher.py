import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import time
import math
from dtaidistance import dtw  # Import DTW
import shutil  # For creating/deleting directories

# --- Configuration ---
# TODO: Replace these with the actual paths to your downloaded video files
# Example paths - update these!
VIDEO_PATH_1 = (
    "match_results/match_Dw9cBXaT0ao_vs_KJkPMbhlMnU_at_030944670/KJkPMbhlMnU_full.mp4"
)
VIDEO_PATH_2 = "match_results/match_Dw9cBXaT0ao_vs_KJkPMbhlMnU_at_030944670/Dw9cBXaT0ao_segment_030944670.mp4"
MATCHED_FRAMES_OUTPUT_DIR = (
    "matched_frames_output"  # Directory to save matched frame pairs
)

# Frame extraction settings
FRAME_SAMPLE_RATE = 1  # Extract 1 frame per second
BATCH_SIZE = 32  # Batch size for embedding generation (adjust based on GPU memory)
# SIMILARITY_THRESHOLD = 0.8 # Threshold might be less relevant with DTW path score

# Model selection
MODEL_NAME = "openai/clip-vit-base-patch32"
# --- End Configuration ---

# --- Global Variables ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
PROCESSOR = None

# --- Helper Functions ---


def load_model():
    """Loads the CLIP model and processor."""
    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None:
        print(f"Loading CLIP model '{MODEL_NAME}' onto {DEVICE}...")
        try:
            MODEL = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
            PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL, PROCESSOR


def extract_frames(video_path, sample_rate=1):
    """
    Extracts frames from a video file at a specified sample rate.

    Args:
        video_path (str or Path): Path to the video file.
        sample_rate (int): Number of frames to extract per second.

    Returns:
        list: A list of PIL Image objects representing the extracted frames.
              Returns an empty list if the video cannot be opened or has no frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return []

    print(f"Extracting frames from '{video_path.name}' at {sample_rate} FPS...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    frames = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not determine video FPS. Assuming 30.")
        fps = 30  # Default fallback

    frame_interval = int(fps / sample_rate) if sample_rate > 0 and fps > 0 else 1
    if frame_interval == 0:
        frame_interval = 1  # Ensure we capture at least some frames

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Sample frame based on interval
        if frame_count % frame_interval == 0:
            # Convert frame from BGR (OpenCV default) to RGB (PIL/CLIP default)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)

        frame_count += 1

    cap.release()
    end_time = time.time()
    print(f"Extracted {len(frames)} frames in {end_time - start_time:.2f} seconds.")
    return frames


def embed_frames_in_batches(frames, model, processor, batch_size=32):
    """
    Generates embeddings for a list of frames using batch processing.

    Args:
        frames (list): A list of PIL Image objects.
        model (CLIPModel): The loaded CLIP model.
        processor (CLIPProcessor): The loaded CLIP processor.
        batch_size (int): Number of frames to process in each batch.

    Returns:
        torch.Tensor: A tensor containing the embeddings for all frames,
                      normalized and moved to the specified device.
                      Returns None if input frames list is empty.
    """
    if not frames:
        print("Warning: No frames provided for embedding.")
        return None

    print(
        f"Generating embeddings for {len(frames)} frames (batch size: {batch_size})..."
    )
    all_embeddings = []
    num_batches = math.ceil(len(frames) / batch_size)
    start_time = time.time()

    with torch.no_grad():  # Disable gradient calculations for inference
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            if not batch_frames:
                continue  # Should not happen with ceil, but safety check

            # Preprocess batch
            inputs = processor(
                text=None, images=batch_frames, return_tensors="pt", padding=True
            )
            inputs = {
                k: v.to(DEVICE) for k, v in inputs.items()
            }  # Move tensors to device

            # Get image features (embeddings)
            image_features = model.get_image_features(**inputs)

            # Normalize embeddings (good practice for cosine similarity)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(
                image_features  # Keep on device for faster similarity calculation
            )

            if (i + 1) % 10 == 0 or (
                i + 1
            ) == num_batches:  # Print progress periodically
                print(f"  Processed batch {i+1}/{num_batches}")

    end_time = time.time()
    print(f"Embedding generation took {end_time - start_time:.2f} seconds.")

    if not all_embeddings:
        return None

    # Concatenate embeddings from all batches
    full_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Generated embeddings shape: {full_embeddings.shape}")
    return full_embeddings  # Keep on DEVICE


def calculate_similarity_matrix(embeds1, embeds2):
    """
    Calculates the cosine similarity matrix between two sets of embeddings.

    Args:
        embeds1 (torch.Tensor): Embeddings for sequence 1 (N x EmbedDim) on DEVICE.
        embeds2 (torch.Tensor): Embeddings for sequence 2 (M x EmbedDim) on DEVICE.

    Returns:
        np.ndarray: An N x M numpy array of cosine similarities.
                    Returns None if input embeddings are invalid.
    """
    if (
        embeds1 is None
        or embeds2 is None
        or embeds1.shape[0] == 0
        or embeds2.shape[0] == 0
    ):
        print("Error: Invalid or empty embeddings provided for similarity calculation.")
        return None

    print(f"Calculating similarity matrix ({embeds1.shape[0]} x {embeds2.shape[0]})...")
    start_time = time.time()

    # Ensure embeddings are on the correct device (should already be, but double-check)
    embeds1 = embeds1.to(DEVICE)
    embeds2 = embeds2.to(DEVICE)

    # Calculate cosine similarity (dot product of normalized embeddings)
    # Resulting shape: (N x M)
    similarity_matrix = torch.matmul(embeds1, embeds2.T)

    end_time = time.time()
    print(f"Similarity matrix calculation took {end_time - start_time:.2f} seconds.")

    # Move to CPU and convert to numpy for DTW library
    return similarity_matrix.cpu().numpy()


def find_dtw_alignment(similarity_matrix, embeds1_np=None, embeds2_np=None):
    """
    Finds the optimal alignment path using DTW.
    It prioritizes using the pre-calculated similarity matrix but may
    fall back to recalculating from embeddings if necessary for path finding
    in the specific dtaidistance version/setup.

    Args:
        similarity_matrix (np.ndarray): Pre-calculated N x M matrix of cosine similarities.
        embeds1_np (np.ndarray, optional): Embeddings for sequence 1 (N x EmbedDim).
                                            Required for some fallback/direct path methods.
        embeds2_np (np.ndarray, optional): Embeddings for sequence 2 (M x EmbedDim).
                                            Required for some fallback/direct path methods.


    Returns:
        tuple: (path, path_avg_similarity)
               path (list): List of tuples (index1, index2) representing the optimal path.
                            Returns None if DTW fails.
               path_avg_similarity (float): Average similarity along the DTW path.
                                            Returns 0.0 if path is None.
    """
    if similarity_matrix is None or similarity_matrix.size == 0:
        print("Error: Cannot perform DTW on invalid similarity matrix.")
        return None, 0.0

    print("Finding optimal alignment path using DTW...")
    start_time = time.time()

    # DTW typically minimizes distance. Convert similarity to distance.
    distance_matrix = 1 - similarity_matrix
    distance_matrix[distance_matrix < 0] = 0  # Ensure non-negative

    path = None
    try:
        # METHOD 1: Try using dtw_from_metric (if available - handles custom distance)
        # This requires the original sequences (embeddings)
        if embeds1_np is not None and embeds2_np is not None:
            try:
                # Define a function for cosine distance (1 - similarity)
                def cosine_dist_func(a, b):
                    # Assuming a and b are single embedding vectors
                    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                    return 1.0 - similarity

                # Use dtw_from_metric if the version supports it well for paths
                # Note: This might recalculate distances, but ensures path finding works
                # Check dtaidistance docs for the exact way to get path from dtw_from_metric
                # For now, let's assume we need warping_paths which might take a metric
                # If not, we rely on the matrix method below.
                # This part is complex, let's stick to matrix if possible.
                pass  # Skip this for now, focus on matrix path finding
            except Exception as e_metric:
                print(f"Info: dtw_from_metric approach failed or not used: {e_metric}")

        # METHOD 2: Try finding path from the distance matrix using internal/C functions
        # This is the most efficient if the function exists and works.
        try:
            from dtaidistance import dtw_cc

            # Use the C implementation's path finding from matrix
            path = dtw_cc.warping_path_from_matrix(distance_matrix.astype(np.double))
            print(
                "Info: Using optimized C implementation (dtw_cc.warping_path_from_matrix)."
            )
        except (ImportError, AttributeError):
            print("Info: dtw_cc.warping_path_from_matrix not found or failed.")
            # METHOD 3: Fallback - Use the standard warping_paths with sequences
            # This is less ideal as it recalculates distances (likely Euclidean by default)
            # unless we can specify cosine distance properly.
            # Let's assume for now the user MUST have the C version working for path_from_matrix.
            # If Method 2 failed, we report error based on that.
            print(
                "Error: Failed to find DTW path using the optimized C function from the distance matrix."
            )
            print("       Ensure dtaidistance C components are correctly installed.")
            return None, 0.0

    except Exception as e:
        # Catch other potential errors during DTW
        print(f"Error during DTW calculation: {e}")
        if "Sequences are too short" in str(e):
            print("  DTW failed likely due to very short frame sequences.")
        return None, 0.0

    end_time = time.time()
    print(f"DTW path calculation took {end_time - start_time:.2f} seconds.")

    if not path:
        print("DTW did not return a valid path.")
        return None, 0.0

    # Calculate average similarity along the found path using the original similarity matrix
    try:
        path_similarities = [similarity_matrix[i, j] for i, j in path]
        path_avg_similarity = np.mean(path_similarities) if path_similarities else 0.0
    except IndexError:
        print(
            "Error: Index out of bounds when accessing similarity matrix with DTW path."
        )
        print(
            f"       Matrix shape: {similarity_matrix.shape}, Max path indices: {max(path, key=lambda x: x[0]) if path else 'N/A'}, {max(path, key=lambda x: x[1]) if path else 'N/A'}"
        )
        return None, 0.0

    print(
        f"DTW path found with {len(path)} points. Average similarity along path: {path_avg_similarity:.4f}"
    )

    return path, path_avg_similarity


def save_matched_frames(
    frames1, frames2, path, output_dir, prefix1="query", prefix2="target"
):
    """
    Saves pairs of frames identified by the DTW path to an output directory.

    Args:
        frames1 (list): List of PIL Image objects for the first video.
        frames2 (list): List of PIL Image objects for the second video.
        path (list): List of tuples (index1, index2) from DTW.
        output_dir (str or Path): Directory to save the matched frame pairs.
        prefix1 (str): Prefix for filenames from the first video.
        prefix2 (str): Prefix for filenames from the second video.
    """
    if not path:
        print("No path provided, skipping frame saving.")
        return

    output_path = Path(output_dir)
    # Clear existing directory or create a new one
    if output_path.exists():
        print(f"Clearing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(path)} matched frame pairs to: {output_path.resolve()}")

    num_digits = len(str(len(path) - 1))  # For zero-padding filenames

    for i, (idx1, idx2) in enumerate(path):
        try:
            frame1 = frames1[idx1]
            frame2 = frames2[idx2]

            # Create filenames with zero-padding
            filename1 = (
                output_path / f"match_{i:0{num_digits}d}_{prefix1}_frame{idx1}.png"
            )
            filename2 = (
                output_path / f"match_{i:0{num_digits}d}_{prefix2}_frame{idx2}.png"
            )

            frame1.save(filename1)
            frame2.save(filename2)

        except IndexError:
            print(
                f"Warning: Index out of bounds for pair {i} (Indices: {idx1}, {idx2}). Skipping."
            )
        except Exception as e:
            print(f"Error saving frame pair {i} (Indices: {idx1}, {idx2}): {e}")

    print("Finished saving matched frames.")


# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Frame-Based Video Matching with DTW ---")
    print(f"Using device: {DEVICE}")

    # Validate video paths
    video1_path = Path(VIDEO_PATH_1)
    video2_path = Path(VIDEO_PATH_2)
    if not video1_path.exists():
        print(f"FATAL ERROR: Video 1 not found at '{VIDEO_PATH_1}'")
        exit()
    if not video2_path.exists():
        print(f"FATAL ERROR: Video 2 not found at '{VIDEO_PATH_2}'")
        exit()

    # 1. Load Model
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"FATAL ERROR: Could not load embedding model. Exiting. Error: {e}")
        exit()

    # 2. Extract Frames (Keep the PIL images)
    frames1_pil = extract_frames(video1_path, FRAME_SAMPLE_RATE)
    frames2_pil = extract_frames(video2_path, FRAME_SAMPLE_RATE)

    if not frames1_pil or not frames2_pil:
        print("FATAL ERROR: Failed to extract frames from one or both videos. Exiting.")
        exit()

    # 3. Generate Embeddings (Keep on DEVICE initially)
    embeddings1_t = embed_frames_in_batches(frames1_pil, model, processor, BATCH_SIZE)
    embeddings2_t = embed_frames_in_batches(frames2_pil, model, processor, BATCH_SIZE)

    if embeddings1_t is None or embeddings2_t is None:
        print("FATAL ERROR: Failed to generate embeddings. Exiting.")
        exit()

    # Keep numpy copies on CPU if needed for fallback methods (optional here)
    # embeds1_np = embeddings1_t.cpu().numpy()
    # embeds2_np = embeddings2_t.cpu().numpy()

    # 4. Calculate Full Similarity Matrix (using tensors on DEVICE -> numpy on CPU)
    similarity_matrix = calculate_similarity_matrix(embeddings1_t, embeddings2_t)

    if similarity_matrix is None:
        print("FATAL ERROR: Failed to calculate similarity matrix. Exiting.")
        exit()

    # 5. Find DTW Alignment Path (using the similarity matrix)
    # Pass the matrix. Pass embeddings only if a fallback method requires them.
    dtw_path, path_avg_sim = find_dtw_alignment(
        similarity_matrix
    )  # , embeds1_np, embeds2_np)

    # 6. Report Results & Save Matched Frames
    print("\n--- DTW Matching Results ---")
    if dtw_path:
        print(
            f"✅ Optimal alignment path found between '{video1_path.name}' and '{video2_path.name}'."
        )
        print(f"   Path Length: {len(dtw_path)} points (matched frame pairs)")
        print(f"   Average Cosine Similarity along Path: {path_avg_sim:.4f}")

        # Save the matched frames
        save_matched_frames(
            frames1_pil,
            frames2_pil,
            dtw_path,
            MATCHED_FRAMES_OUTPUT_DIR,
            prefix1=video1_path.stem,  # Use video filename stem as prefix
            prefix2=video2_path.stem,
        )
    else:
        print(
            f"❌ No suitable DTW alignment path found between '{video1_path.name}' and '{video2_path.name}'."
        )
        # Reasons for failure might include DTW errors or very dissimilar videos.

    print("\n--- Frame Matching Process Finished ---")
