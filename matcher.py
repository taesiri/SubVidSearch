import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os  # Added for file existence check
import pickle  # Added
import time  # Added for timing cache operations
from pathlib import Path  # Added

# Load the model once when the module is imported
# This avoids reloading the model every time a match is performed
print("Loading sentence transformer model...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# --- Constants ---
MODEL_NAME = "all-MiniLM-L6-v2"
# Consider making BATCH_SIZE configurable or adjusting based on available memory
BATCH_SIZE = 64
# Minimum similarity score to consider a match valid (adjust as needed)
MIN_SIMILARITY_THRESHOLD = 0.65  # Example threshold

# --- Helper Functions ---


def _parse_timestamp(ts_str):
    """Helper to parse VTT timestamp string (HH:MM:SS.ms)"""
    # Basic parsing, can be made more robust if needed
    try:
        h, m, s_ms = ts_str.split(":")
        s, ms = s_ms.split(".")
        return int(h), int(m), int(s), int(ms)
    except ValueError:
        # Handle cases like '0:00:01.123' if they occur
        parts = ts_str.split(":")
        if len(parts) == 2:  # MM:SS.ms
            m, s_ms = parts
            h = 0
            s, ms = s_ms.split(".")
            return int(h), int(m), int(s), int(ms)
        # Add more robust parsing if needed
        print(f"Warning: Could not parse timestamp: {ts_str}")
        return 0, 0, 0, 0


def load_transcript_blocks_with_timestamps(vtt_filepath):
    """
    Loads transcript text blocks with start and end timestamps from a VTT file.
    Returns a list of dictionaries: [{'start': str, 'end': str, 'text': str}].
    """
    if not os.path.exists(vtt_filepath):
        print(f"Error: VTT file not found at {vtt_filepath}")
        return []

    try:
        with open(vtt_filepath, "r", encoding="utf-8") as f:
            transcript_text = f.read()
    except Exception as e:
        print(f"Error reading VTT file {vtt_filepath}: {e}")
        return []

    blocks = []
    current_text = []
    current_start_time = None
    current_end_time = None

    lines = transcript_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if (
            not line
            or line == "WEBVTT"
            or line.startswith("STYLE")
            or line.startswith("REGION")
            or line.startswith("NOTE")
        ):
            continue

        time_match = re.match(
            r"^(?:\d+:)?\d{2}:\d{2}\.\d{3}\s+-->\s+(?:\d+:)?\d{2}:\d{2}\.\d{3}", line
        )
        if time_match:
            if current_text and current_start_time is not None:
                blocks.append(
                    {
                        "start": current_start_time,
                        "end": current_end_time,
                        "text": " ".join(current_text),
                    }
                )
                current_text = []

            # Extract times directly from the matched line
            parts = line.split(" --> ")
            current_start_time = parts[0]
            # Handle potential cue settings after end timestamp
            current_end_time = parts[1].split(" ")[0]

            while i < len(lines) and lines[i].strip():
                text_line = lines[i].strip()
                if not re.match(r"^[a-zA-Z]+:", text_line):
                    current_text.append(text_line)
                i += 1
            continue

        if current_start_time is None:
            continue

    if current_text and current_start_time is not None:
        blocks.append(
            {
                "start": current_start_time,
                "end": current_end_time,
                "text": " ".join(current_text),
            }
        )

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
    similarity_matrix = util.cos_sim(clip_embeds, window_embeds).cpu().numpy()
    # Average the diagonal elements (similarity of clip_block_1 vs window_block_1, etc.)
    return similarity_matrix.diagonal().mean()


def get_cache_path(subtitle_path):
    """Generates the cache file path based on the subtitle file path."""
    p = Path(subtitle_path)
    return p.with_suffix(".pkl")


def load_embeddings_from_cache(cache_path):
    """Loads blocks (as list of dicts) and embeddings from a cache file."""
    if not cache_path.exists():
        return None
    try:
        print(f"[Matcher] Loading embeddings from cache: {cache_path.name}")
        start_time = time.time()
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        end_time = time.time()
        print(f"[Matcher] Cache loaded in {end_time - start_time:.2f} seconds.")

        # Validate format: tuple(list[dict], np.ndarray)
        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[0], list)
            and isinstance(data[1], np.ndarray)
        ):
            # Check if the first element (blocks) is a list of dicts with expected keys
            if not data[0] or (
                isinstance(data[0][0], dict)
                and all(k in data[0][0] for k in ["start", "end", "text"])
            ):
                # Check if embedding dimensions match model output
                # Note: This assumes MODEL is loaded and accessible here.
                # If MODEL isn't guaranteed, this check might need adjustment.
                if (
                    data[1].shape[0] == len(data[0])
                    and data[1].shape[1] == MODEL.get_sentence_embedding_dimension()
                ):
                    print("[Matcher] Cache format validated.")
                    return data[0], data[1]  # Return blocks, embeddings
                else:
                    print(
                        f"[Matcher] Warning: Cache file {cache_path.name} has embedding dimension mismatch or block count mismatch. Ignoring cache."
                    )
                    return None
            else:
                print(
                    f"[Matcher] Warning: Cache file {cache_path.name} has unexpected block structure (expected list of dicts). Ignoring cache."
                )
                return None
        else:
            print(
                f"[Matcher] Warning: Cache file {cache_path.name} has unexpected format (expected tuple(list, ndarray)). Ignoring cache."
            )
            return None
    except (
        pickle.UnpicklingError,
        EOFError,
        TypeError,
        ValueError,
        AttributeError,
        ModuleNotFoundError,
        ImportError,
    ) as e:  # Added ModuleNotFoundError/ImportError for custom classes if used later
        print(
            f"[Matcher] Error loading cache file {cache_path.name}: {e}. Recomputing embeddings."
        )
        # Optionally delete the corrupted cache file: cache_path.unlink(missing_ok=True)
        return None
    except Exception as e:
        print(
            f"[Matcher] An unexpected error occurred loading cache {cache_path.name}: {e}"
        )
        return None


def save_embeddings_to_cache(cache_path, blocks, embeddings):
    """Saves blocks (list of dicts) and embeddings to a cache file."""
    try:
        print(f"[Matcher] Attempting to save embeddings to cache: {cache_path}")
        start_time = time.time()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure blocks are in the expected dict format before saving
        if not blocks or not isinstance(blocks[0], dict):
            print(
                "[Matcher] Error: Attempting to save blocks in incorrect format. Aborting cache save."
            )
            return

        print(f"[Matcher] Cache directory exists: {cache_path.parent.exists()}")
        print(
            f"[Matcher] Saving {len(blocks)} blocks and embeddings shape {embeddings.shape}..."
        )

        with open(cache_path, "wb") as f:
            print("[Matcher] Opened cache file for writing...")
            pickle.dump(
                (blocks, embeddings), f
            )  # Save as tuple (list[dict], np.ndarray)
            print("[Matcher] pickle.dump completed.")

        end_time = time.time()
        # Check if file exists immediately after saving
        if cache_path.exists():
            print(f"[Matcher] Cache file successfully created: {cache_path}")
            print(f"[Matcher] Cache saved in {end_time - start_time:.2f} seconds.")
        else:
            print(
                f"[Matcher] *** Error: Cache file NOT found after saving attempt: {cache_path} ***"
            )

    except Exception as e:
        # Print the specific exception
        print(f"[Matcher] *** Error saving cache file {cache_path}: {e} ***")
        import traceback

        traceback.print_exc()


def find_best_match(full_vtt_path, clip_vtt_path, window_size_factor=1.0):
    """
    Finds the best matching segment of the full transcript for the clip transcript.
    Utilizes caching for the full video's embeddings.

    Args:
        full_vtt_path (str): Path to the VTT file of the long video.
        clip_vtt_path (str): Path to the VTT file of the short clip.
        window_size_factor (float): Multiplies the clip length to determine
                                    the search window size in the full transcript.

    Returns:
        tuple: (start_timestamp, end_timestamp, similarity_score, start_index, matched_preview_text)
               Returns (None, None, 0, -1, "") if matching fails or inputs are invalid.
    """
    print(
        f"\nMatching '{os.path.basename(clip_vtt_path)}' against '{os.path.basename(full_vtt_path)}'..."
    )

    # --- Handle Full Transcript: Load from Cache or Compute ---
    full_blocks = None
    full_embeds = None
    full_cache_path = get_cache_path(full_vtt_path)
    print(f"[Matcher] Cache path determined for full VTT: {full_cache_path}")

    cached_data = load_embeddings_from_cache(full_cache_path)
    if cached_data:
        full_blocks, full_embeds = cached_data
        print(
            f"[Matcher] Using cached embeddings for {os.path.basename(full_vtt_path)}"
        )
    else:
        print(
            f"[Matcher] No valid cache found for {os.path.basename(full_vtt_path)}. Processing VTT..."
        )
        full_blocks = load_transcript_blocks_with_timestamps(full_vtt_path)
        if full_blocks:
            full_texts = [block["text"] for block in full_blocks]
            if full_texts:
                print(f"Embedding {len(full_texts)} full blocks...")
                full_embeds = embed_blocks(full_texts, MODEL)
                # Save to cache if embeddings were successful
                if full_embeds.shape[0] > 0:
                    save_embeddings_to_cache(full_cache_path, full_blocks, full_embeds)
            else:
                print("[Matcher] No text found in full transcript blocks.")
                full_embeds = np.array([])  # Ensure embeds is an empty array
        else:
            print("[Matcher] Failed to load blocks from full transcript VTT.")
            full_embeds = np.array([])  # Ensure embeds is an empty array

    # --- Handle Clip Transcript: Load and Embed (No Caching for Clip) ---
    clip_blocks = load_transcript_blocks_with_timestamps(clip_vtt_path)

    # --- Validation and Setup ---
    if (
        full_blocks is None
        or not clip_blocks
        or full_embeds is None
        or full_embeds.shape[0] == 0
    ):
        print(
            "Matching failed: Could not load or embed transcript blocks for full video or clip."
        )
        return None, None, 0, -1, ""

    clip_len = len(clip_blocks)
    full_len = len(full_blocks)  # Use length of loaded blocks

    if clip_len == 0:
        print("Matching failed: Clip transcript has no blocks.")
        return None, None, 0, -1, ""

    window_size = int(clip_len * window_size_factor)
    window_size = max(clip_len, min(window_size, full_len))

    if clip_len > full_len:
        print(
            "Warning: Clip transcript is longer than the full transcript. Cannot match."
        )
        return None, None, 0, -1, ""
    # Window size adjustment warning already handled by min(window_size, full_len)

    clip_texts = [block["text"] for block in clip_blocks]
    if not clip_texts:
        print("Matching failed: No text found in clip blocks.")
        return None, None, 0, -1, ""

    print(f"Embedding {len(clip_texts)} clip blocks...")
    clip_embeds = embed_blocks(clip_texts, MODEL)

    if clip_embeds.shape[0] == 0:
        print("Matching failed: Could not generate clip embeddings.")
        return None, None, 0, -1, ""

    # --- Perform Matching ---
    best_index = -1
    best_score = -1.0

    print(
        f"Performing sliding window match (window size: {window_size}, clip length: {clip_len})..."
    )
    # Ensure loop range is correct even if window_size adjusted to full_len
    for i in range(
        full_len - clip_len + 1
    ):  # Iterate up to the last possible start for the clip
        # Extract the window embeddings from the full transcript
        # We only need the segment corresponding to the clip length for direct comparison
        window_segment_embeds = full_embeds[i : i + clip_len]

        # Calculate average cosine similarity
        score = average_cosine_similarity(clip_embeds, window_segment_embeds)

        if score > best_score:
            best_score = score
            best_index = i

    # --- Process Results ---
    if (
        best_index != -1 and best_score >= MIN_SIMILARITY_THRESHOLD
    ):  # Added threshold check
        # Get timestamps and text from the block dictionaries
        start_timestamp = full_blocks[best_index]["start"]
        # Ensure the end index doesn't go out of bounds
        end_block_index = min(best_index + clip_len - 1, full_len - 1)
        end_timestamp = full_blocks[end_block_index]["end"]

        match_preview = " ".join(
            block["text"] for block in full_blocks[best_index : best_index + clip_len]
        )
        print(f"Match found: Index {best_index}, Score {best_score:.4f}")
        return start_timestamp, end_timestamp, best_score, best_index, match_preview
    elif best_index != -1:
        print(
            f"Potential match found (Index {best_index}, Score {best_score:.4f}) but below threshold ({MIN_SIMILARITY_THRESHOLD})."
        )
        return None, None, 0, -1, ""
    else:
        print("No suitable match found.")
        return None, None, 0, -1, ""
