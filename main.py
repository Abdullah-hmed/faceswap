from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import mimetypes
import argparse

from faceswap.image import ImageSwapper
from faceswap.video import VideoSwapper

# Argument parser for command line options
parser = argparse.ArgumentParser(description="Face Swap with InsightFace and Inswapper_128")
parser.add_argument("-f", "--face", help="Path to the face to be used")
parser.add_argument("-m", "--media", help="Path to target image (face to be replaced)")
parser.add_argument("-o", "--output", default="output.png", help="Output image path (default: output.png)")
args = parser.parse_args()

# Init face detector
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

# Load inswapper model
swapper = get_model('models/inswapper_128.onnx', download=False, download_zip=False)

# Check if the input is an image or video based on MIME type
face_mime_type, _ = mimetypes.guess_type(args.face)
media_mime_type, _ = mimetypes.guess_type(args.media)

if face_mime_type.startswith('image') and media_mime_type:

    if media_mime_type.startswith('image'):
        ImageSwapper(app, swapper, args.face, args.media, args.output).swap_faces()
    elif media_mime_type.startswith('video'):
        VideoSwapper(app, swapper, args.face, args.media, args.output).swap_faces()
    else:
        ValueError(f"Unsupported media type: {media_mime_type}")

else:
    ValueError("Please provide valid image paths for both face and media.")

