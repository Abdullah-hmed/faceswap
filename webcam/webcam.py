import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
from PIL import Image
import io
import cv2, os
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from utils.helpers import highres_swap

detector = FaceAnalysis(name='buffalo_l', root='models', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
detector.prepare(ctx_id=0, det_size=(320, 320))  # Prepare the face detector with a specific context ID and detection size

swapper = get_model('models/inswapper_128.onnx', download=False, download_zip=False)  # Load the face swapper model


app = Flask(__name__)
# Allow all origins for development. Restrict this in production for security.
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    """
    Renders the main HTML page.
    """
    face_files = os.listdir('faces')  # List available faces in the 'faces' directory
    face_list = [f for f in face_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]  # Filter for image files
    return render_template('index.html', face_list=face_list)

@socketio.on('connect')
def handle_connect():
    """
    Handles client connection event.
    """
    print(f'Client connected: {request.sid}')
    emit('status', {'message': 'Connected to server!'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handles client disconnection event.
    """
    print(f'Client disconnected: {request.sid}')

@socketio.on('frame') # Changed from 'image' to 'frame' to match frontend
def handle_frame(data):
    """
    Handles incoming webcam frames from the client.
    Echoes the received image back along with dummy prediction data.
    """
    image_data_base64 = data['image'] # Extract the image data from the received object
    upscale_value = data.get('upscale_value', 1)
    face_swap = data.get('face_swap', None) # Optional face swap parameter
    try:
        # --- Placeholder for your ML model inference or image processing ---
        # In a real application, you would process image_data_base64 here.
        # For this request, we are simply echoing it back.
        image = Image.open(io.BytesIO(base64.b64decode(image_data_base64.split(',')[1]))) # Decode base64 and open image
        # Load the source image for swapping (replace 'source.jpg' with your actual source image filename)
        source_path = f'faces/{face_swap}'
        source_img = cv2.imread(source_path)
        target_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect faces
        src_faces = detector.get(source_img)
        tgt_faces = detector.get(target_img)

        # Swap first face found if both images have faces
        if src_faces and tgt_faces:
            swapped = swapper.get(target_img, tgt_faces[0], src_faces[0], paste_back=True)
            swapped = highres_swap(swapper, target_img, tgt_faces[0], src_faces[0], upscale=upscale_value)
            # Convert swapped image back to PIL Image for encoding
            swapped_rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
            swapped_pil = Image.fromarray(swapped_rgb)
            buffered = io.BytesIO()
            swapped_pil.save(buffered, format="JPEG")
            swapped_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
            image_data_base64 = swapped_base64
        # Simulate a prediction result
        dummy_label = "ECHO" # Example label
        dummy_confidence = 0.75 # Example confidence

        # Emit the prediction and the received image back to the client
        emit('frame_response', {
            'image': image_data_base64 # Send the received image back
        }, room=request.sid)

    except Exception as e:
        print(f"Error processing frame from client {request.sid}: {e}")
        emit('error', {'message': f'Server error processing frame: {e}'}, room=request.sid)

if __name__ == '__main__':
    print("--- Starting Webcam Swapper Flask Socket.IO Server ---")
    print("Server will be accessible on http://0.0.0.0:3000")
    print("To connect, open your HTML client file in a web browser.")
    # Set debug=True for development, allow_unsafe_werkzeug=True for non-production environments
    socketio.run(app, host='0.0.0.0', port=3000, debug=True, allow_unsafe_werkzeug=True)
