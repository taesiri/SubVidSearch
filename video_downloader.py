import yt_dlp
import subprocess
from pathlib import Path
from datetime import timedelta
import os
import time
import random


def timestamp_to_seconds(ts_str):
    """Converts HH:MM:SS.ms timestamp string to total seconds (float)."""
    if not isinstance(ts_str, str):
        print(f"Warning: Invalid timestamp format '{ts_str}'. Expected string.")
        return 0.0
    try:
        h, m, s_ms = ts_str.split(":")
        s, ms = s_ms.split(".")
        total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        return total_seconds
    except ValueError:
        print(f"Warning: Could not parse timestamp '{ts_str}'. Returning 0.0 seconds.")
        return 0.0


def get_direct_video_audio_urls(video_url):
    """Gets direct URLs for the best MP4 video and M4A audio streams, avoiding manifests."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        # Prioritize non-manifest MP4 video + M4A audio
        # Filter out hls (m3u8) and dash (mpd) protocols explicitly
        "format": (
            "bestvideo[ext=mp4][protocol!*=m3u8][protocol!*=dash]+"
            "bestaudio[ext=m4a][protocol!*=m3u8][protocol!*=dash]/"
            "best[ext=mp4][protocol!*=m3u8][protocol!*=dash]/"  # Fallback to best combined MP4 (non-manifest)
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"  # Fallback allowing manifests if absolutely necessary
            "best[ext=mp4]/best"  # Final fallback
        ),
        "cookiesfrombrowser": ("firefox", None),
    }
    video_stream_url = None
    audio_stream_url = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            requested_formats = info.get("requested_formats")

            # Debug: Print info about formats found
            # print(f"Debug yt-dlp info: {info}") # Uncomment for deep debugging if needed
            # print(f"Debug yt-dlp requested_formats: {requested_formats}")

            if requested_formats:
                # Check if the selected formats are manifests
                video_fmt = requested_formats[0]
                is_video_manifest = "m3u8" in video_fmt.get(
                    "protocol", ""
                ) or "dash" in video_fmt.get("protocol", "")
                print(
                    f"Debug: Requested video format protocol: {video_fmt.get('protocol')}, URL: {video_fmt.get('url')[:100]}..."
                )  # Debug

                if not is_video_manifest:
                    video_stream_url = video_fmt.get("url")
                    if len(requested_formats) > 1:
                        audio_fmt = requested_formats[1]
                        is_audio_manifest = "m3u8" in audio_fmt.get(
                            "protocol", ""
                        ) or "dash" in audio_fmt.get("protocol", "")
                        print(
                            f"Debug: Requested audio format protocol: {audio_fmt.get('protocol')}, URL: {audio_fmt.get('url')[:100]}..."
                        )  # Debug
                        if not is_audio_manifest:
                            audio_stream_url = audio_fmt.get("url")
                        else:
                            print(
                                "Warning: yt-dlp selected a manifest for audio, trying fallback."
                            )
                            audio_stream_url = (
                                None  # Force fallback search for direct audio
                            )
                    elif video_fmt.get("acodec") != "none":
                        # Combined format, but check if it's a manifest
                        if not is_video_manifest:
                            # Let ffmpeg handle audio from single source if it's not a manifest
                            audio_stream_url = None
                            print("Debug: Using combined, non-manifest format.")
                        else:
                            print(
                                "Warning: yt-dlp selected a combined manifest format, trying fallback."
                            )
                            video_stream_url = None  # Force fallback search
                    else:  # Video only, need separate audio
                        audio_stream_url = (
                            None  # Force fallback search for direct audio
                        )

                else:  # Video is a manifest, force fallback search
                    print(
                        "Warning: yt-dlp selected a manifest for video, trying fallback."
                    )
                    video_stream_url = None

            # Fallback if requested_formats didn't work or gave manifests
            if not video_stream_url:
                print(
                    "Debug: Falling back to manual format search (avoiding manifests)."
                )
                formats = info.get("formats", [])
                # Prefer non-manifest MP4 video
                video_formats = [
                    f
                    for f in formats
                    if f.get("vcodec") != "none"
                    and f.get("ext") == "mp4"
                    and "m3u8" not in f.get("protocol", "")
                    and "dash" not in f.get("protocol", "")
                    and f.get("url")  # Ensure URL exists
                ]
                # Prefer non-manifest M4A audio
                audio_formats = [
                    f
                    for f in formats
                    if f.get("acodec") != "none"
                    and f.get("ext") == "m4a"
                    and "m3u8" not in f.get("protocol", "")
                    and "dash" not in f.get("protocol", "")
                    and f.get("url")  # Ensure URL exists
                ]

                if video_formats:
                    video_formats.sort(
                        key=lambda f: (f.get("height", 0), f.get("tbr", 0)),
                        reverse=True,
                    )
                    video_stream_url = video_formats[0].get("url")
                    print(
                        f"Debug Fallback: Selected video format {video_formats[0].get('format_id')}, protocol {video_formats[0].get('protocol')}"
                    )
                if audio_formats:
                    audio_formats.sort(key=lambda f: f.get("abr", 0), reverse=True)
                    audio_stream_url = audio_formats[0].get("url")
                    print(
                        f"Debug Fallback: Selected audio format {audio_formats[0].get('format_id')}, protocol {audio_formats[0].get('protocol')}"
                    )
                elif not audio_stream_url and video_stream_url:
                    # Check if the selected video format has audio
                    selected_video_format_info = next(
                        (f for f in video_formats if f.get("url") == video_stream_url),
                        None,
                    )
                    if (
                        selected_video_format_info
                        and selected_video_format_info.get("acodec") != "none"
                    ):
                        print(
                            "Debug Fallback: Selected video format includes audio. Setting audio_stream_url to None."
                        )
                        audio_stream_url = None  # Use combined stream
                    else:
                        print(
                            "Debug Fallback: No separate M4A audio found, and video format lacks audio."
                        )

            if not video_stream_url:
                print(
                    f"Warning: Could not find suitable *direct* video stream URL for {video_url}"
                )
            if video_stream_url and not audio_stream_url:
                # Double check if we selected a combined format that has audio
                all_formats = info.get("formats", [])
                selected_video_format_info = next(
                    (f for f in all_formats if f.get("url") == video_stream_url), None
                )
                if (
                    selected_video_format_info
                    and selected_video_format_info.get("acodec") != "none"
                ):
                    print(
                        "Debug: Confirmed selected video URL has audio. Setting audio_stream_url to None for ffmpeg."
                    )
                    audio_stream_url = (
                        None  # Ensure ffmpeg uses the single input for audio
                    )

            return video_stream_url, audio_stream_url

    except Exception as e:
        print(f"Error getting direct URLs for {video_url}: {e}")
        # import traceback # Uncomment for detailed exception traceback
        # traceback.print_exc() # Uncomment for detailed exception traceback
        return None, None


def download_segment(
    video_stream_url, audio_stream_url, start_time_str, end_time_str, output_path: Path
):
    """Downloads a specific segment using ffmpeg with improved settings."""
    # 1. Convert the *exact* provided start/end timestamps to seconds
    start_seconds = timestamp_to_seconds(start_time_str)
    end_seconds = timestamp_to_seconds(end_time_str)
    # 2. Calculate the *exact* duration based on the provided timestamps
    duration_seconds = end_seconds - start_seconds

    if duration_seconds <= 0:
        print(
            f"Error: Invalid segment duration ({duration_seconds}s) for {output_path.name}. Start: {start_time_str}, End: {end_time_str}"
        )
        return False

    # --- Debugging: Print Input URLs ---
    print(f"Debug: Video URL for ffmpeg: {video_stream_url}")
    if audio_stream_url:
        print(f"Debug: Audio URL for ffmpeg: {audio_stream_url}")
    else:
        print("Debug: No separate audio URL provided.")
    # --- End Debugging ---

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary filename during download
    temp_filename = f"temp_{random.randint(1000,9999)}_{output_path.name}"
    temp_output_path = output_path.with_name(temp_filename)

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        # "-loglevel", # Temporarily disable quiet logging for debugging
        # "error",
        "-loglevel",  # Use verbose logging
        "verbose",  # Change to verbose for detailed ffmpeg output
        "-y",  # Overwrite output files without asking
    ]

    # 3. Use the *exact* calculated start_seconds for seeking (-ss)
    ffmpeg_cmd.extend(["-ss", str(start_seconds)])
    ffmpeg_cmd.extend(["-i", video_stream_url])

    if audio_stream_url:
        # Apply the same exact start_seconds to the audio input
        ffmpeg_cmd.extend(["-ss", str(start_seconds)])
        ffmpeg_cmd.extend(["-i", audio_stream_url])

    # 4. Use the *exact* calculated duration_seconds for the segment length (-t)
    ffmpeg_cmd.extend(["-t", str(duration_seconds)])

    if audio_stream_url:
        # Map video from first input (0), audio from second input (1)
        ffmpeg_cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    else:
        # Map video and audio (if present) from the single input (0)
        # '?' makes the audio mapping optional, preventing errors if no audio stream exists
        ffmpeg_cmd.extend(["-map", "0:v:0", "-map", "0:a:0?"])

    # Output settings based on the example (higher quality)
    ffmpeg_cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "slow",  # Slower preset for better quality/compression
            "-crf",
            "17",  # Lower CRF for higher visual quality (17-18 often considered visually lossless)
            "-profile:v",
            "high",  # Use high profile for better quality
            "-level",
            "5.1",  # Support for higher resolutions like 4K
            "-movflags",
            "+faststart",  # Optimize for web streaming
            "-c:a",
            "aac",
            "-b:a",
            "320k",  # Higher audio bitrate
            "-ar",
            "48000",  # Higher audio sample rate
            str(temp_output_path),  # Output to temporary file first
        ]
    )

    print(f"Running FFmpeg for segment: {output_path.name}")
    # --- Debugging: Print Full Command ---
    # Use list comprehension for safer quoting if paths contain spaces, though less likely with URLs
    cmd_string_for_print = " ".join(
        f'"{arg}"' if " " in arg else arg for arg in ffmpeg_cmd
    )
    print(f"Debug: FFmpeg command: {cmd_string_for_print}")
    # --- End Debugging ---
    try:
        process = subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,  # Keep timeout
        )

        # --- Debugging: Print FFmpeg output even on success ---
        print(f"Debug: FFmpeg return code: {process.returncode}")
        print(f"Debug: FFmpeg stdout:\n{process.stdout}")
        print(f"Debug: FFmpeg stderr:\n{process.stderr}")
        # --- End Debugging ---

        # Check if temp file exists and is not empty, then rename
        temp_exists = temp_output_path.exists()
        temp_size = temp_output_path.stat().st_size if temp_exists else 0

        if temp_exists and temp_size > 0:
            print(
                f"Debug: Temporary file {temp_output_path} exists (Size: {temp_size}). Renaming."
            )  # Debug
            os.rename(temp_output_path, output_path)
            print(f"Successfully downloaded segment: {output_path.name}")
            return True
        else:
            # --- Debugging: Check temp file status on failure ---
            if not temp_exists:
                print(
                    f"Debug: Temporary file {temp_output_path} does not exist after ffmpeg success code."
                )
            else:
                print(
                    f"Debug: Temporary file {temp_output_path} exists but size is {temp_size}."
                )
            # --- End Debugging ---
            print(
                f"Error: FFmpeg ran successfully (RC: {process.returncode}) but temporary output file is missing or empty: {temp_output_path.name}"
            )
            # Stderr was already printed in debug block above
            # Clean up empty/failed temp file if it exists
            if temp_exists:
                try:
                    temp_output_path.unlink()
                    print(
                        f"Debug: Removed empty/failed temp file {temp_output_path}"
                    )  # Debug
                except OSError as e:
                    print(
                        f"Warning: Could not remove temp file {temp_output_path}: {e}"
                    )
            return False

    except subprocess.TimeoutExpired:
        print(f"Error: FFmpeg timed out downloading segment {output_path.name}")
        # Clean up temp file on timeout
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
                print(
                    f"Debug: Removed temp file {temp_output_path} after timeout"
                )  # Debug
            except OSError as e:
                print(
                    f"Warning: Could not remove temp file {temp_output_path} after timeout: {e}"
                )
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error downloading segment {output_path.name} with FFmpeg.")
        print(f"Return code: {e.returncode}")
        # --- Debugging: Ensure stdout/stderr are printed ---
        print(f"FFmpeg stdout: {e.stdout}")  # Print stdout too
        print(f"FFmpeg stderr: {e.stderr}")
        # --- End Debugging ---
        # Clean up temp file on error
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
                print(
                    f"Debug: Removed temp file {temp_output_path} after error"
                )  # Debug
            except OSError as unlink_e:
                print(
                    f"Warning: Could not remove temp file {temp_output_path} after error: {unlink_e}"
                )
        return False
    except Exception as e:
        print(
            f"An unexpected error occurred during segment download {output_path.name}: {e}"
        )
        # Clean up temp file on unexpected error
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
                print(
                    f"Debug: Removed temp file {temp_output_path} after unexpected error"
                )  # Debug
            except OSError as unlink_e:
                print(
                    f"Warning: Could not remove temp file {temp_output_path} after error: {unlink_e}"
                )
        return False


def download_full_video(video_url, output_path: Path):
    """Downloads the full video using yt-dlp."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_path),
        "quiet": False,
        "no_warnings": True,
        "progress_hooks": [lambda d: print(f"yt-dlp status: {d['status']}", end="\r")],
        "cookiesfrombrowser": ("firefox", None),  # Optional
        "retries": 3,  # Add retries
        "fragment_retries": 3,
    }
    print(f"\nDownloading full video: {output_path.name}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        # Check if file exists and is not empty after download
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"\nSuccessfully downloaded full video: {output_path.name}")
            return True
        else:
            print(
                f"\nError: yt-dlp finished but output file is missing or empty: {output_path.name}"
            )
            return False
    except Exception as e:
        print(f"\nError downloading full video {output_path.name}: {e}")
        # Clean up potentially incomplete file
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as unlink_e:
                print(
                    f"Warning: Could not remove potentially incomplete file {output_path}: {unlink_e}"
                )
        return False
