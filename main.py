from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import mimetypes
import argparse

from faceswap.image import ImageSwapper
from faceswap.video import VideoSwapper
from utils.helpers import extract_faces_from_img, display_faces_terminal
import os, sys
from contextlib import redirect_stdout, redirect_stderr


def main():
    parser = argparse.ArgumentParser(description="Face Swap with InsightFace and Inswapper_128")
    parser.add_argument("-f", "--face", help="Path to the face to be used")
    parser.add_argument("-m", "--media", help="Path to target image (face to be replaced)")
    parser.add_argument("-o", "--output", default="output.png", help="Output image path (default: output.png)")
    parser.add_argument("--target-face", help="Path to the target face image for similarity matching", default=None)
    parser.add_argument('--choose-face', action='store_true', help='Choose a face to swap from within image')

    args = parser.parse_args()

    # Check if the input is an image or video based on MIME type
    face_mime_type, _ = mimetypes.guess_type(args.face)
    media_mime_type, _ = mimetypes.guess_type(args.media)

    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0)
            swapper = get_model('models/inswapper_128.onnx', download=False, download_zip=False)


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
            if args.choose_face:

                faces_list = extract_faces_from_img(app, args.media)
                display_faces_terminal(faces_list, len(faces_list))
                try:
                    chosen_face_idx = int(input("Which face would you like to swap? "))
                except ValueError:
                    print("❌ Please enter a valid number.")
                ImageSwapper(app, swapper, args.face, args.media, args.output, target_face_embedding=faces_list[chosen_face_idx]['embedding']).swap_faces()

            else:
                ImageSwapper(app, swapper, args.face, args.media, args.output).swap_faces()

        elif media_mime_type.startswith('video'):
            VideoSwapper(app, swapper, args.face, args.media, args.output, target_face_path=args.target_face, similarity_threshold=0.15).swap_faces()
        else:
            ValueError(f"Unsupported media type: {media_mime_type}")

    else:
        ValueError("Please provide valid image paths for both face and media.")

if __name__ == "__main__":
    main()