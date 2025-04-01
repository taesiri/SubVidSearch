import os
import downloader
import matcher

# --- Configuration ---
SUBTITLES_DIR = "subtitles"
LONG_VIDEO_URL = "https://www.youtube.com/watch?v=Dw9cBXaT0ao"  # Example Long Video
SHORT_VIDEOS_URLS = [
    "https://www.youtube.com/watch?v=KJkPMbhlMnU",  # Example Short Clip 1
]
# --- End Configuration ---


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
        return  # Exit if we don't have the main transcript

    print(f"[Downloader] Long video subtitle ready: {long_video_subtitle_path}")

    # 2. Download subtitles for short videos and match each one
    print(f"\n[Downloader] Processing {len(SHORT_VIDEOS_URLS)} short video(s)...")
    successful_clips = []
    for i, url in enumerate(SHORT_VIDEOS_URLS):
        print(f"\n--- Processing Short Video {i+1}: {url} ---")
        clip_subtitle_path = downloader.download_subtitle(url, SUBTITLES_DIR)

        if clip_subtitle_path:
            print(f"[Downloader] Short video subtitle ready: {clip_subtitle_path}")
            successful_clips.append({"url": url, "path": clip_subtitle_path})
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
    for clip_info in successful_clips:
        clip_path = clip_info["path"]
        clip_url = clip_info["url"]

        # Use the matcher module function - unpack the new end_timestamp
        start_timestamp, end_timestamp, score, index, preview = matcher.find_best_match(
            long_video_subtitle_path, clip_path
        )

        # Check if start_timestamp is not None (indicates a successful match)
        if start_timestamp:
            print("\n✅ Match Found!")
            print(f"   Clip URL: {clip_url}")
            # Print the time range
            print(
                f"   Matched Time Range in Long Video: {start_timestamp} --> {end_timestamp}"
            )
            print(f"   Similarity Score: {score:.4f}")
            print(f"   Start Block Index in Long Video: {index}")
            print(f'   Matched Text Preview:\n   """\n   {preview}\n   """')
            match_results.append(
                {
                    "clip_url": clip_url,
                    "start_timestamp": start_timestamp,  # Store start
                    "end_timestamp": end_timestamp,  # Store end
                    "score": score,
                    "index": index,
                }
            )
        else:
            print(f"\n❌ No match found for clip: {clip_url}")

    print("\n--- Matching Process Finished ---")
    if match_results:
        print("Summary of Matches Found:")
        for result in match_results:
            # Update summary print
            print(
                f"- Clip: {result['clip_url']} -> Time Range: {result['start_timestamp']} --> {result['end_timestamp']} (Score: {result['score']:.4f})"
            )
    else:
        print("No matches were found for any of the short clips.")


if __name__ == "__main__":
    run()
