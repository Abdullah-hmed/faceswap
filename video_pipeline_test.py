from utils.i2v import frames_to_video
from utils.v2i import video_to_frames
from utils.audio_util import extract_audio, add_audio_to_video
import sys, os, shutil
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from faceswap.image import ImageSwapper
from tqdm import tqdm

providers = onnxruntime.get_available_providers()

# Init face detector
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

# Load inswapper model
swapper = get_model('models/inswapper_128.onnx', download=False, download_zip=False)

def swap_faces_frames(frame_path, face_source_path, app, swapper):
    frames = sorted(os.listdir(frame_path))  # ensure correct order
    if not frames:
        print("❌ No frames found in the directory.")
        return

    swapped_dir = "swapped"
    os.makedirs(swapped_dir, exist_ok=True)

    for frame in tqdm(frames, desc="Swapping faces"):
        full_frame_path = os.path.join(frame_path, frame)
        output_path = os.path.join("swapped", f"swapped_{frame}")

        swapper_instance = ImageSwapper(app, swapper, face_source_path, full_frame_path, output_path)
        swapper_instance.swap_faces()

    return swapped_dir


def test_video_to_frames(video_path, output_name, face):
    frame_path, fps = video_to_frames(video_path)
    
    if not os.path.isdir(frame_path):
        print(f"❌ Frame path not found: {frame_path}")
        return
    
    swapped_dir = swap_faces_frames(frame_path, face, app, swapper)

    frames_to_video(swapped_dir, output_name, fps=fps)
    print(f"🎞️  Video frames extracted and saved as {output_name}")
    return swapped_dir, frame_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <video_path> <new_video_output_name> <face>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_video = sys.argv[2]
    face = sys.argv[3]
    temp_audio = "temp_audio.aac"

    extract_audio(video_path, temp_audio)
    swapped_directory, frame_directory = test_video_to_frames(video_path, output_video, face) # Extract frames and create video
    add_audio_to_video(output_video, temp_audio, output_video)

    # Optional: clean up temporary audio
    if os.path.exists(temp_audio) and os.path.isdir(frame_directory) and os.path.exists(swapped_directory):
        print("Cleaning up temporary files.")
        os.remove(temp_audio)
        shutil.rmtree(frame_directory)
        shutil.rmtree(swapped_directory)
        print("✅ Temporary files cleaned up.")

    sys.exit(0)