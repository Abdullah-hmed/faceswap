import os
import cv2
import shutil
from tqdm import tqdm

# Import the utility functions from your working code's imports
from utils.i2v import frames_to_video
from utils.v2i import video_to_frames
from utils.audio_util import extract_audio, add_audio_to_video

from utils.helpers import find_most_similar_face, find_most_similar_face_with_score, highres_swap

app = None
swapper = None
source_face = None # This will be set inside the VideoSwapper class now, or passed in
compare_face = None # This will be set inside the VideoSwapper class now, or passed in
class VideoSwapper:
    def __init__(self, app_instance, swapper_model, face_path, video_path, output_path="output.mp4", upscale=1, compare_face_embedding=None, similarity_threshold=0.1):
        self.app = app_instance
        self.swapper = swapper_model
        self.face_path = face_path
        self.video_path = video_path
        self.output_path = output_path
        self.upscale = upscale
        self.compare_face_embedding = compare_face_embedding
        self.similarity_threshold = similarity_threshold

        self.base_name = os.path.splitext(os.path.basename(video_path))[0]
        # These temp paths are now managed by the utils functions
        self.frames_dir = None # Will be returned by video_to_frames
        self.audio_path = f"temp_audio_{self.base_name}.aac" # Unique temp audio file

        # Pre-load the source face once for efficiency, similar to init_worker's purpose
        source_img = cv2.imread(self.face_path)
        faces = self.app.get(source_img)
        if not faces:
            raise RuntimeError(f"No face found in source image at {face_path}!")
        
        self.compare_face_embedding = compare_face_embedding

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

        swapped_dir = f"{self.base_name}_swapped_frames"
        os.makedirs(swapped_dir, exist_ok=True)

        for frame_name in tqdm(frames, desc="🧠 Swapping faces in frames"):
            full_frame_path = os.path.join(self.frames_dir, frame_name)
            output_frame_path = os.path.join(swapped_dir, f"swapped_{frame_name}")

            target_img = cv2.imread(full_frame_path)
            target_faces = self.app.get(target_img)

            if not target_faces:
                cv2.imwrite(output_frame_path, target_img)
                continue

            current_frame_with_swaps = target_img.copy()

            if self.compare_face_embedding is not None:
                best_face, score = find_most_similar_face_with_score(target_faces, self.compare_face_embedding)


                if best_face and best_face.gender == self.source_face.gender and score >= self.similarity_threshold:
                    # swapped_img = self.swapper.get(current_frame_with_swaps, best_face, self.source_face, paste_back=True)
                    swapped_img = highres_swap(self.swapper, current_frame_with_swaps, best_face, self.source_face, upscale=self.upscale)
                else:
                    # No suitable match found — save original
                    swapped_img = None
            else:
                # No compare_face set — loop through
                swapped_img = current_frame_with_swaps
                for face in target_faces:
                    # if face.gender != self.source_face.gender:
                    #     continue
                    # swapped_img = self.swapper.get(swapped_img, face, self.source_face, paste_back=True)
                    swapped_img = highres_swap(self.swapper, swapped_img, face, self.source_face, upscale=self.upscale)

            if swapped_img is not None:
                cv2.imwrite(output_frame_path, swapped_img)
            else:
                # print(f"⚠️ No suitable face to swap in frame: {frame_name}. Saving original.")
                cv2.imwrite(output_frame_path, target_img)
            
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