import os
import cv2
import shutil
from tqdm import tqdm

# Import the utility functions from your working code's imports
from utils.i2v import frames_to_video
from utils.v2i import video_to_frames
from utils.audio_util import extract_audio, add_audio_to_video

app = None
swapper = None
source_face = None # This will be set inside the VideoSwapper class now, or passed in

class VideoSwapper:
    def __init__(self, app_instance, swapper_model, face_path, video_path, output_path="output.mp4"):
        self.app = app_instance
        self.swapper = swapper_model
        self.face_path = face_path
        self.video_path = video_path
        self.output_path = output_path

        self.base_name = os.path.splitext(os.path.basename(video_path))[0]
        # These temp paths are now managed by the utils functions
        self.frames_dir = None # Will be returned by video_to_frames
        self.audio_path = f"temp_audio_{self.base_name}.aac" # Unique temp audio file

        # Pre-load the source face once for efficiency, similar to init_worker's purpose
        source_img = cv2.imread(self.face_path)
        faces = self.app.get(source_img)
        if not faces:
            raise RuntimeError(f"No face found in source image at {face_path}!")
        self.source_face = faces[0]

    def _swap_faces_in_frames(self):
        """
        Processes frames from self.frames_dir, swaps faces, and saves them
        to a new 'swapped' directory.
        """
        frames = sorted(os.listdir(self.frames_dir))
        if not frames:
            print("❌ No frames found in the directory for swapping.")
            return None

        swapped_dir = f"{self.base_name}_swapped_frames" # Create a unique swapped dir
        os.makedirs(swapped_dir, exist_ok=True)

        for frame_name in tqdm(frames, desc="🧠 Swapping faces in frames"):
            full_frame_path = os.path.join(self.frames_dir, frame_name)
            output_frame_path = os.path.join(swapped_dir, f"swapped_{frame_name}")

            target_img = cv2.imread(full_frame_path)
            
            target_faces = self.app.get(target_img)
            
            # If no faces are found in the target frame, just copy the original frame
            if not target_faces:
                cv2.imwrite(output_frame_path, target_img)
                continue

            # Swap faces for each detected face in the target frame
            current_frame_with_swaps = target_img.copy() # Work on a copy
            for face in target_faces:
                # Use swapper directly to perform the swap
                current_frame_with_swaps = self.swapper.get(current_frame_with_swaps, face, self.source_face, paste_back=True)
            
            # Save the swapped frame
            cv2.imwrite(output_frame_path, current_frame_with_swaps)
            
        return swapped_dir


    def swap_faces(self):
        print(f"📽️ Processing video: {self.video_path}")

        # 1. Extract Audio
        print("🔊 Extracting audio from the video...")
        try:
            extract_audio(self.video_path, self.audio_path)
            has_audio = True
        except Exception as e:
            print(f"❌ Could not extract audio: {e}. Proceeding without audio.")
            has_audio = False

        # 2. Extract Frames
        print("🖼️ Extracting video frames...")
        self.frames_dir, fps = video_to_frames(self.video_path)
        if not os.path.isdir(self.frames_dir):
            print(f"❌ Frame extraction failed or directory not found: {self.frames_dir}")
            return

        # 3. Swap Faces in Frames
        swapped_frames_dir = self._swap_faces_in_frames()
        if not swapped_frames_dir:
            print("❌ Face swapping in frames failed.")
            return

        # 4. Rebuild Video from Swapped Frames
        print("🎞️ Rebuilding video from swapped frames...")
        # frames_to_video expects the directory containing the swapped frames and the output name
        frames_to_video(swapped_frames_dir, self.output_path, fps=fps)

        # 5. Merge Audio back (if extracted)
        if has_audio and os.path.exists(self.audio_path) and os.path.exists(self.output_path):
            print("🔊 Merging audio back into the new video...")
            # add_audio_to_video expects input video, audio, and output video path
            add_audio_to_video(self.output_path, self.audio_path, self.output_path)
        elif has_audio and not os.path.exists(self.audio_path):
            print("⚠️ Audio was supposed to be extracted but temp audio file not found. Skipping audio merge.")
        else:
            print("❌ Skipping audio merge as no audio was found in the original video.")

        # 6. Clean up temporary files
        print("🧹 Cleaning up temporary files...")
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)
        if os.path.isdir(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        if os.path.isdir(swapped_frames_dir):
            shutil.rmtree(swapped_frames_dir)
        print("✅ Temporary files cleaned up.")

        print(f"✅ Done! Final video saved at: {self.output_path}")
        return self.output_path