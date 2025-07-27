import cv2
from utils.helpers import find_most_similar_face

class ImageSwapper:
    def __init__(self, app, swapper, source_path, target_path, output_path, target_face_path=None):
        self.app = app
        self.swapper = swapper
        self.source_path = source_path
        self.target_path = target_path
        self.output_path = output_path
        self.target_face_path = target_face_path

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
            if not self.target_face_path:
                print("⚠️ Multiple faces found in target image, but no reference face provided. Using first face by default.")
                selected_tgt_face = tgt_faces[0]
            else:
                print(f"🔍 Matching target face using: {self.target_face_path}")
                compare_img = cv2.imread(self.target_face_path)

                if compare_img is None:
                    raise ValueError(f"❌ Could not load reference face image at {self.target_face_path}")

                compare_faces = self.app.get(compare_img)

                if not compare_faces:
                    raise ValueError("❌ No face detected in reference image.")
                if len(compare_faces) > 1:
                    raise ValueError("❌ Multiple faces detected in reference image. Please provide an image with only one face.")

                selected_tgt_face = find_most_similar_face(tgt_faces, compare_faces[0])
        
        # Perform face swap
        result = self.swapper.get(target_img, selected_tgt_face, src_faces[0], paste_back=True)

        if result is not None:
            cv2.imwrite(self.output_path, result)
            print(f"✅ Face swapped image saved at: {self.output_path}")
        else:
            raise ValueError("❌ Face swap failed. Please check the images and try again.")