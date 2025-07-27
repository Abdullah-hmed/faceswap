from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import mimetypes
import argparse

from faceswap.image import ImageSwapper
from faceswap.video import VideoSwapper
import os, sys, contextlib

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def main():
    parser = argparse.ArgumentParser(description="Face Swap with InsightFace and Inswapper_128")
    parser.add_argument("-f", "--face", help="Path to the face to be used")
    parser.add_argument("-m", "--media", help="Path to target image (face to be replaced)")
    parser.add_argument("-o", "--output", default="output.png", help="Output image path (default: output.png)")
    parser.add_argument("--target-face", help="Path to the target face image for similarity matching", default=None)

    args = parser.parse_args()

    with suppress_stderr():
        # Init face detector
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0)

    # Load inswapper model
    swapper = get_model('models/inswapper_128.onnx', download=False, download_zip=False)

    # Check if the input is an image or video based on MIME type
    face_mime_type, _ = mimetypes.guess_type(args.face)
    media_mime_type, _ = mimetypes.guess_type(args.media)

    # Setting output name if not provided
    if not args.output or args.output == "output.png":
        face_base = os.path.splitext(os.path.basename(args.face))[0]
        media_base = os.path.splitext(os.path.basename(args.media))[0]
        args.output = f"{face_base}_{media_base}"
        
        if media_mime_type and media_mime_type.startswith('image'):
            args.output += ".png"
        elif media_mime_type and media_mime_type.startswith('video'):
            args.output += ".mp4"

    if face_mime_type.startswith('image') and media_mime_type:

        if media_mime_type.startswith('image'):
            ImageSwapper(app, swapper, args.face, args.media, args.output, target_face_path=args.target_face).swap_faces()
        elif media_mime_type.startswith('video'):
            VideoSwapper(app, swapper, args.face, args.media, args.output, target_face_path=args.target_face).swap_faces()
        else:
            ValueError(f"Unsupported media type: {media_mime_type}")

    else:
        ValueError("Please provide valid image paths for both face and media.")

if __name__ == "__main__":
    main()