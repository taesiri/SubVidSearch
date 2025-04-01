import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os  # Added for file existence check

# Load the model once when the module is imported
# This avoids reloading the model every time a match is performed
print("Loading sentence transformer model...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def load_transcript_blocks_with_timestamps(vtt_filepath):
    """Loads transcript text blocks with start timestamps from a VTT file."""
    if not os.path.exists(vtt_filepath):
        print(f"Error: VTT file not found at {vtt_filepath}")
        return []  # Return empty list if file doesn't exist

    try:
        with open(vtt_filepath, "r", encoding="utf-8") as f:
            transcript_text = f.read()
    except Exception as e:
        print(f"Error reading VTT file {vtt_filepath}: {e}")
        return []  # Return empty list on read error

    blocks = []
    current_text = []
    current_time = None

    # Improved regex to handle potential variations and ignore WEBVTT header/styles
    lines = transcript_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Skip empty lines, WEBVTT header, STYLE, REGION etc.
        if (
            not line
            or line == "WEBVTT"
            or line.startswith("STYLE")
            or line.startswith("REGION")
            or line.startswith("NOTE")
        ):
            continue

        # Look for timestamp lines
        time_match = re.match(
            r"^(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}", line
        )
        if time_match:
            # If we have accumulated text from the previous block, save it
            if current_text and current_time is not None:
                blocks.append((current_time, " ".join(current_text)))
                current_text = []  # Reset for the new block

            current_time = time_match.group(1)  # Store the start time

            # Read subsequent lines as text content until the next timestamp or empty line
            while i < len(lines) and lines[i].strip():
                text_line = lines[i].strip()
                # Ignore VTT cue settings like align:start position:15%
                if not re.match(r"^[a-zA-Z]+:", text_line):
                    current_text.append(text_line)
                i += 1
            continue  # Move to the next line after processing text block

        # If it's not a timestamp line and we haven't found the first timestamp yet, skip
        if current_time is None:
            continue

        # This part should ideally not be reached if parsing logic is correct,
        # but handles potential stray text lines not associated with a timestamp block
        # if line: # Append stray lines if needed, though VTT format usually groups them
        #    current_text.append(line)

    # Add the last block if any text was collected
    if current_text and current_time is not None:
        blocks.append((current_time, " ".join(current_text)))

    if not blocks:
        print(f"Warning: No valid timestamped blocks found in {vtt_filepath}")

    return blocks


def embed_blocks(text_blocks, model):
    """Generates embeddings for a list of text blocks."""
    if not text_blocks:
        return np.array([])
    embeddings = model.encode(text_blocks, convert_to_tensor=True)
    return embeddings.cpu().numpy()


def average_cosine_similarity(clip_embeds, window_embeds):
    """Calculates the average diagonal cosine similarity between two sets of embeddings."""
    # Ensure embeddings are not empty and shapes match for diagonal calculation
    if clip_embeds.shape[0] == 0 or window_embeds.shape[0] == 0:
        return 0.0
    if clip_embeds.shape[0] != window_embeds.shape[0]:
        # This shouldn't happen with the sliding window logic, but good to check
        print(
            f"Warning: Shape mismatch for cosine similarity ({clip_embeds.shape[0]} vs {window_embeds.shape[0]})"
        )
        # Fallback: Calculate full similarity matrix and average? Or return 0?
        # For sliding window, a mismatch indicates an issue elsewhere. Let's return 0.
        return 0.0

    # Calculate cosine similarity for corresponding blocks
    similarity_matrix = cosine_similarity(clip_embeds, window_embeds)
    # Average the diagonal elements (similarity of clip_block_1 vs window_block_1, etc.)
    return similarity_matrix.diagonal().mean()


def find_best_match(full_vtt_path, clip_vtt_path, window_size_factor=1.0):
    """
    Finds the best matching segment of the full transcript for the clip transcript.

    Args:
        full_vtt_path (str): Path to the VTT file of the long video.
        clip_vtt_path (str): Path to the VTT file of the short clip.
        window_size_factor (float): Multiplies the clip length to determine
                                    the search window size in the full transcript.
                                    1.0 means the window size is exactly the clip length.
                                    Can be adjusted slightly > 1.0 if needed.

    Returns:
        tuple: (start_timestamp, similarity_score, start_index, matched_preview_text)
               Returns (None, 0, -1, "") if matching fails or inputs are invalid.
    """
    print(
        f"\nMatching '{os.path.basename(clip_vtt_path)}' against '{os.path.basename(full_vtt_path)}'..."
    )

    full_blocks = load_transcript_blocks_with_timestamps(full_vtt_path)
    clip_blocks = load_transcript_blocks_with_timestamps(clip_vtt_path)

    if not full_blocks or not clip_blocks:
        print("Matching failed: Could not load transcript blocks.")
        return None, 0, -1, ""

    clip_len = len(clip_blocks)
    full_len = len(full_blocks)
    window_size = int(clip_len * window_size_factor)
    # Ensure window size isn't larger than the full transcript or smaller than the clip
    window_size = max(clip_len, min(window_size, full_len))

    if clip_len > full_len:
        print(
            "Warning: Clip transcript is longer than the full transcript. Cannot match."
        )
        return None, 0, -1, ""
    if window_size > full_len:
        print(
            f"Adjusted window size ({window_size}) is larger than full transcript length ({full_len}). Setting window size to clip length ({clip_len})."
        )
        window_size = (
            clip_len  # Fallback to exact clip length if factor makes it too large
        )

    full_texts = [text for _, text in full_blocks]
    clip_texts = [text for _, text in clip_blocks]

    # Only embed if texts are available
    if not clip_texts or not full_texts:
        print("Matching failed: No text found in blocks.")
        return None, 0, -1, ""

    print(
        f"Embedding {len(clip_texts)} clip blocks and {len(full_texts)} full blocks..."
    )
    clip_embeds = embed_blocks(clip_texts, MODEL)
    full_embeds = embed_blocks(full_texts, MODEL)

    if clip_embeds.shape[0] == 0 or full_embeds.shape[0] == 0:
        print("Matching failed: Could not generate embeddings.")
        return None, 0, -1, ""

    best_index = -1
    best_score = -1.0  # Initialize with a value lower than possible cosine similarity

    print(
        f"Performing sliding window match (window size: {window_size}, clip length: {clip_len})..."
    )
    # Iterate through the full transcript embeddings with the sliding window
    # The loop should go up to full_len - window_size + 1
    for i in range(full_len - window_size + 1):
        # Extract the window embeddings from the full transcript
        window_embeds = full_embeds[i : i + window_size]

        # We need to compare the *clip* embeddings against the *window* embeddings.
        # If window_size is exactly clip_len, we compare directly.
        # If window_size > clip_len (due to factor), we might need a strategy:
        #   Option A: Compare clip_embeds against the first clip_len embeds in the window.
        #   Option B: Calculate similarity differently (e.g., max similarity within window).
        # Let's stick to Option A for simplicity, assuming window_size is primarily for context
        # and the core match is based on the clip's length.
        current_window_segment_embeds = window_embeds[
            :clip_len
        ]  # Take the part matching clip length

        # Calculate average cosine similarity between the clip and the current window segment
        score = average_cosine_similarity(clip_embeds, current_window_segment_embeds)

        if score > best_score:
            best_score = score
            best_index = i  # The start index of the best matching window

    if best_index != -1:
        matched_timestamp, _ = full_blocks[best_index]
        # Preview text should correspond to the actual matched segment length (clip_len)
        match_preview = " ".join(
            text for _, text in full_blocks[best_index : best_index + clip_len]
        )
        print(f"Match found: Index {best_index}, Score {best_score:.4f}")
        return matched_timestamp, best_score, best_index, match_preview
    else:
        print("No suitable match found.")
        return None, 0, -1, ""
