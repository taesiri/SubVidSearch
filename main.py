import os
import subtitle_downloader as downloader
import matcher
import video_downloader
from pathlib import Path
import json
from datetime import datetime

# --- Configuration ---
SUBTITLES_DIR = "subtitles"
LONG_VIDEO_URL = "https://www.youtube.com/watch?v=Dw9cBXaT0ao"  # Example Long Video
SHORT_VIDEOS_URLS = [
    "https://www.youtube.com/watch?v=KJkPMbhlMnU",  # Example Short Clip 1
]
RESULTS_DIR = "match_results"  # Directory to save downloaded clips/segments
# --- End Configuration ---

# --- Helper Functions (Keep metadata saving here for now) ---


def save_match_metadata(metadata, filepath):
    """Saves match metadata to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"Saved metadata to: {filepath.name}")
    except Exception as e:
        print(f"Error saving metadata to {filepath}: {e}")


# --- Main Execution Function ---


def run():
    """Main execution function."""
    print("--- Starting Video Subtitle Matching Process ---")
    print(f"Subtitle directory: '{SUBTITLES_DIR}'")

    # 1. Download subtitle for the long video
    print(f"\n[Downloader] Processing long video: {LONG_VIDEO_URL}")
    long_video_subtitle_path = downloader.download_subtitle(
        LONG_VIDEO_URL, SUBTITLES_DIR
    )

    if not long_video_subtitle_path:
        print(
            "\nError: Failed to download or find subtitle for the long video. Cannot proceed with matching."
        )
        return

    print(f"[Downloader] Long video subtitle ready: {long_video_subtitle_path}")
    long_video_id = downloader.extract_video_id(LONG_VIDEO_URL)

    # 2. Download subtitles for short videos and match each one
    print(f"\n[Downloader] Processing {len(SHORT_VIDEOS_URLS)} short video(s)...")
    successful_clips = []
    for i, url in enumerate(SHORT_VIDEOS_URLS):
        print(f"\n--- Processing Short Video {i+1}: {url} ---")
        clip_subtitle_path = downloader.download_subtitle(url, SUBTITLES_DIR)

        if clip_subtitle_path:
            print(f"[Downloader] Short video subtitle ready: {clip_subtitle_path}")
            clip_video_id = downloader.extract_video_id(url)
            if clip_video_id:
                successful_clips.append(
                    {"url": url, "path": clip_subtitle_path, "id": clip_video_id}
                )
            else:
                print(
                    f"Warning: Could not extract video ID for clip {url}. Skipping download steps if matched."
                )
                successful_clips.append(
                    {"url": url, "path": clip_subtitle_path, "id": None}
                )
        else:
            print(f"[Downloader] Failed to get subtitle for short video: {url}")

    # 3. Perform matching for successfully downloaded clips
    print(f"\n--- Starting Matching Process ---")
    if not successful_clips:
        print(
            "No short video subtitles were successfully downloaded. Nothing to match."
        )
        return

    match_results = []
    downloaded_files_info = []

    base_results_dir = Path(RESULTS_DIR)
    base_results_dir.mkdir(parents=True, exist_ok=True)

    for clip_info in successful_clips:
        clip_path = clip_info["path"]
        clip_url = clip_info["url"]
        clip_id = clip_info["id"]

        # Use the matcher module function
        start_timestamp, end_timestamp, score, index, preview = matcher.find_best_match(
            long_video_subtitle_path, clip_path
        )

        if start_timestamp and long_video_id and clip_id:
            print("\n✅ Match Found!")
            print(f"   Clip URL: {clip_url} (ID: {clip_id})")
            print(f"   Long Video URL: {LONG_VIDEO_URL} (ID: {long_video_id})")
            print(
                f"   Matched Time Range in Long Video: {start_timestamp} --> {end_timestamp}"
            )
            print(f"   Similarity Score: {score:.4f}")
            print(f"   Start Block Index in Long Video: {index}")

            # --- Download Segment and Clip ---
            timestamp_str_for_dir = start_timestamp.replace(":", "").replace(".", "")
            match_dir = (
                base_results_dir
                / f"match_{long_video_id}_vs_{clip_id}_at_{timestamp_str_for_dir}"
            )
            match_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n   Saving results to: {match_dir}")

            clip_download_path = match_dir / f"{clip_id}_full.mp4"
            segment_download_path = (
                match_dir / f"{long_video_id}_segment_{timestamp_str_for_dir}.mp4"
            )
            metadata_path = match_dir / "match_metadata.json"

            # Download the full short clip using the new module
            clip_download_success = video_downloader.download_full_video(
                clip_url, clip_download_path
            )

            # Download the segment from the long video using the new module
            segment_download_success = False
            long_video_stream_url, long_audio_stream_url = (
                video_downloader.get_direct_video_audio_urls(LONG_VIDEO_URL)
            )
            if long_video_stream_url:
                segment_download_success = video_downloader.download_segment(
                    long_video_stream_url,
                    long_audio_stream_url,
                    start_timestamp,
                    end_timestamp,
                    segment_download_path,
                )
            else:
                print(
                    f"   Skipping segment download for {long_video_id} due to missing stream URL."
                )

            # --- Save Metadata ---
            match_data = {
                "match_timestamp": datetime.now().isoformat(),
                "long_video_url": LONG_VIDEO_URL,
                "long_video_id": long_video_id,
                "clip_video_url": clip_url,
                "clip_video_id": clip_id,
                "match_start_timestamp": start_timestamp,
                "match_end_timestamp": end_timestamp,
                "match_similarity_score": score,
                "match_start_index": index,
                "clip_subtitle_file": str(Path(clip_path).resolve()),
                "long_video_subtitle_file": str(
                    Path(long_video_subtitle_path).resolve()
                ),
                "downloaded_clip_path": (
                    str(clip_download_path.resolve()) if clip_download_success else None
                ),
                "downloaded_segment_path": (
                    str(segment_download_path.resolve())
                    if segment_download_success
                    else None
                ),
                "clip_download_success": clip_download_success,
                "segment_download_success": segment_download_success,
            }
            save_match_metadata(match_data, metadata_path)

            match_results.append(match_data)
            if clip_download_success:
                downloaded_files_info.append(f"Clip: {clip_download_path.name}")
            if segment_download_success:
                downloaded_files_info.append(f"Segment: {segment_download_path.name}")

        elif start_timestamp:
            print("\n✅ Match Found (but missing video IDs for download)!")
            print(f"   Clip URL: {clip_url}")
            print(
                f"   Matched Time Range in Long Video: {start_timestamp} --> {end_timestamp}"
            )
            print(f"   Similarity Score: {score:.4f}")
            match_results.append(
                {
                    "clip_url": clip_url,
                    "start_timestamp": start_timestamp,  # Renamed for consistency below
                    "end_timestamp": end_timestamp,  # Renamed for consistency below
                    "score": score,
                    "index": index,
                    "downloaded_clip_path": None,
                    "downloaded_segment_path": None,
                }
            )

        else:
            print(f"\n❌ No match found for clip: {clip_url}")

    # --- Final Summary ---
    print("\n--- Matching and Download Process Finished ---")
    if match_results:
        print("Summary of Matches Found:")
        for result in match_results:
            # Adjust keys based on whether download happened or not
            clip_vid_url = result.get("clip_video_url", result.get("clip_url"))
            start_ts = result.get(
                "match_start_timestamp", result.get("start_timestamp")
            )
            end_ts = result.get("match_end_timestamp", result.get("end_timestamp"))
            sim_score = result.get("match_similarity_score", result.get("score"))

            status = "Download Skipped/Failed"
            if result.get("downloaded_clip_path") and result.get(
                "downloaded_segment_path"
            ):
                status = "OK"
            elif result.get("downloaded_clip_path") or result.get(
                "downloaded_segment_path"
            ):
                status = "Partial Download"
            elif "downloaded_clip_path" in result:  # Check if download was attempted
                status = "Download Failed"

            print(
                f"- Clip: {clip_vid_url} -> Time: {start_ts} --> {end_ts} (Score: {sim_score:.4f}) [Status: {status}]"
            )
        print(
            "\nDownloaded files saved in subdirectories under:",
            Path(RESULTS_DIR).resolve(),
        )
    else:
        print("No matches were found for any of the short clips.")


if __name__ == "__main__":
    run()
