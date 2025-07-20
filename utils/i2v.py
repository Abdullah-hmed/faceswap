import cv2
import os
import sys
from natsort import natsorted

def frames_to_video(image_folder, output_video, fps=30):
    if not os.path.isdir(image_folder):
        print(f"Folder not found: {image_folder}")
        return

    images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No image files found in the folder.")
        return

    images = natsorted(images)  # Natural sort (frame_0001, frame_0002, ...)
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(image_folder, img_name))
        video.write(frame)

    video.release()
    print(f"Video saved as: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python frames_to_video.py <image_folder> <output_video> [fps]")
        sys.exit(1)

    image_folder = sys.argv[1]
    output_video = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    frames_to_video(image_folder, output_video, fps)
