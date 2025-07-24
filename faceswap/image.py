import cv2

class ImageSwapper:
    def __init__(self, app, swapper, source, target, output):
        self.app = app
        self.swapper = swapper
        self.source = source
        self.target = target
        self.output = output

    def swap_faces(self):
        # Load images
        source_img = cv2.imread(self.source)
        target_img = cv2.imread(self.target)

        # Detect faces
        src_faces = self.app.get(source_img)
        tgt_faces = self.app.get(target_img)

        # Swap first face found
        if src_faces and tgt_faces:
            result = self.swapper.get(target_img, tgt_faces[0], src_faces[0], paste_back=True)
            cv2.imwrite(self.output, result)
            # print(f"✅ Face swapped saved as {self.output}")
        else:
            cv2.imwrite(self.output, target_img)
            # print(f"⚠️ No face(s) found. Original image saved as {self.output}")
            return target_img, self.output