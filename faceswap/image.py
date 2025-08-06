import cv2
import numpy as np
from utils.helpers import cosine_similarity, highres_swap

class ImageSwapper:
    def __init__(self, app, swapper, source_path, target_path, output_path, upscale, target_face_embedding=None):
        self.app = app
        self.swapper = swapper
        self.source_path = source_path
        self.target_path = target_path
        self.output_path = output_path
        self.upscale = upscale
        self.target_face_embedding = target_face_embedding  # Now expects a 512-dim embedding

    def swap_faces(self):
        # Load source and target images
        source_img = cv2.imread(self.source_path)
        target_img = cv2.imread(self.target_path)

        if source_img is None or target_img is None:
            raise ValueError("❌ Failed to load source or target image.")

        # Detect faces
        src_faces = self.app.get(source_img)
        tgt_faces = self.app.get(target_img)

        if not src_faces:
            raise ValueError("❌ No faces detected in source image.")
        if not tgt_faces:
            raise ValueError("❌ No faces detected in target image.")

        # Select the target face
        if len(tgt_faces) == 1:
            selected_tgt_face = tgt_faces[0]
        else:
            if self.target_face_embedding is None:
                print("⚠️ Multiple faces found in target image, but no embedding provided. Using first face by default.")
                selected_tgt_face = tgt_faces[0]
            else:
                print("🔍 Matching target face using provided embedding...")
                best_face = None
                best_score = -1

                for face in tgt_faces:
                    sim = cosine_similarity(face.normed_embedding, self.target_face_embedding)
                    if sim > best_score:
                        best_score = sim
                        best_face = face

                if best_face is None:
                    raise ValueError("❌ Failed to find a matching face based on embedding.")

                selected_tgt_face = best_face

        # Perform face swap
        # result = self.swapper.get(target_img, selected_tgt_face, src_faces[0], paste_back=True)
        result = highres_swap(self.swapper, target_img, selected_tgt_face, src_faces[0], upscale=self.upscale)

        if result is not None:
            cv2.imwrite(self.output_path, result)
            print(f"✅ Face swapped image saved at: {self.output_path}")
        else:
            raise ValueError("❌ Face swap failed. Please check the images and try again.")