import cv2
import os
import sys

def video_to_frames(video_path):
    if not os.path.isfile(video_path):
        print(f"❌ File not found: {video_path}")
        return None, None

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = f"{video_name}_frames"
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"✅ Extracted {frame_count} frames at {fps:.2f} FPS to folder: {output_folder}")
    return output_folder, fps

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_to_frames.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    video_to_frames(video_file)
