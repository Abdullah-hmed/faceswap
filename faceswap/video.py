from faceswap.image import ImageSwapper

class VideoSwapper:
    def __init__(self, app, swapper, face_path, video_path, output_path):
        self.app = app
        self.swapper = swapper
        self.face_path = face_path
        self.video_path = video_path
        self.output_path = output_path

    def swap_faces(self):
        print(f"Processing video: {self.video_path}")
        