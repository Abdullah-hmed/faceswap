import sys
import os
import cv2
import subprocess
from natsort import natsorted

def frames_to_video(image_folder, output_video, fps=30, reencode_h264=True, delete_temp=True):
    if not os.path.isdir(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return

    images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("❌ No image files found in the folder.")
        return

    images = natsorted(images)
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_frame.shape

    temp_output = output_video if not reencode_h264 else output_video.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(image_folder, img_name))
        video.write(frame)

    video.release()
    print(f"✅ Initial video saved as: {temp_output}")

    if reencode_h264:
        print("🔄 Re-encoding to H.264...")
        subprocess.call([
            "ffmpeg", "-y", 
            "-loglevel", "error",
            "-i", temp_output,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "slow",
            "-crf", "23",
            "-movflags", "+faststart",
            output_video
        ], stdin=subprocess.DEVNULL)
        print(f"✅ H.264 re-encoded video saved as: {output_video}")

        if delete_temp and os.path.exists(temp_output):
            os.remove(temp_output)
            print(f"🗑️ Raw file deleted: {temp_output}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python frames_to_video.py <image_folder> <output_video> [fps]")
        sys.exit(1)

    image_folder = sys.argv[1]
    output_video = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    frames_to_video(image_folder, output_video, fps)
