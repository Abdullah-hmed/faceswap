import subprocess
import os

def extract_audio(video_path: str, output_audio_path: str):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                   # Skip video
            "-acodec", "copy",        # Encode audio as AAC
            output_audio_path
        ], check=True)
        print(f"✅ Audio extracted to: {output_audio_path}")
    except subprocess.CalledProcessError:
        print("❌ Failed to extract audio. Does the input video have an audio track?")

def add_audio_to_video(video_path: str, audio_path: str, output_path: str):
    temp_output = f"{output_path}.temp.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            temp_output
        ], check=True)
        os.replace(temp_output, output_path)
        print(f"✅ Audio added to video: {output_path}")
    except subprocess.CalledProcessError:
        print("❌ Failed to add audio to video.")
        if os.path.exists(temp_output):
            os.remove(temp_output)