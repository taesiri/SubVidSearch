import yt_dlp
import os
import re


def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/|v\/|youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    print(f"Warning: Could not extract video ID from URL: {url}")
    return None


def download_subtitle(url, subtitles_dir):
    """
    Download subtitle for a YouTube video if it doesn't already exist.
    Returns the path to the subtitle file or None if download failed or no subtitles exist.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return None

    os.makedirs(subtitles_dir, exist_ok=True)
    base_filepath = os.path.join(subtitles_dir, video_id)
    # Define potential filenames (yt-dlp might use .en.vtt or just .vtt for auto)
    possible_filenames = [
        f"{base_filepath}.en.vtt",
        f"{base_filepath}.vtt",  # Sometimes auto-subs might not have language code
    ]

    # Check if subtitle already exists
    for filename in possible_filenames:
        if os.path.exists(filename):
            print(
                f"Subtitle for {video_id} already exists ({filename}), skipping download."
            )
            return filename

    # Configure yt-dlp options
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,  # Attempt to get auto-generated if manual aren't available
        "subtitleslangs": ["en"],  # Prioritize English
        "subtitlesformat": "vtt",  # Specify VTT format
        "outtmpl": base_filepath,  # Output template (yt-dlp adds .lang.vtt or just .vtt)
        "quiet": True,
        "no_warnings": True,
        "cookiesfrombrowser": ("firefox", None),  # Optional
    }

    print(f"Attempting to download subtitle for {video_id} ({url})...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # Check info first
            # Check if subtitles are available before attempting download
            available_subs = info_dict.get("subtitles", {}) or info_dict.get(
                "automatic_captions", {}
            )
            if not available_subs or "en" not in available_subs:
                print(f"No English subtitles found for video {video_id} ({url}).")
                return None

            # Proceed with download if subtitles seem available
            ydl.download([url])

        # Verify download by checking possible filenames again
        for filename in possible_filenames:
            if os.path.exists(filename):
                print(f"Successfully downloaded subtitle: {filename}")
                return filename

        # If not found via expected names, check directory listing as a fallback
        found_files = [
            f
            for f in os.listdir(subtitles_dir)
            if f.startswith(video_id) and f.endswith(".vtt")
        ]
        if found_files:
            actual_filename = os.path.join(subtitles_dir, found_files[0])
            print(
                f"Successfully downloaded subtitle (found as {found_files[0]}): {actual_filename}"
            )
            return actual_filename
        else:
            # This case might happen if download failed silently or yt-dlp behavior changed
            print(
                f"Subtitle file for {video_id} not found after download attempt, though subtitles were expected."
            )
            return None

    except yt_dlp.utils.DownloadError as e:
        # Check specific error messages
        if (
            "subtitles not available" in str(e).lower()
            or "no subtitles were found" in str(e).lower()
        ):
            print(f"No English subtitles available for video {video_id} ({url}).")
        elif "Video unavailable" in str(e):
            print(f"Video {video_id} ({url}) is unavailable.")
        else:
            print(f"Error downloading subtitle for {video_id} ({url}): {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {video_id} ({url}): {e}")
        return None
