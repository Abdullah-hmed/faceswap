import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich_pixels import Pixels
from PIL import Image
import cv2, shutil
from sklearn.cluster import DBSCAN
from collections import defaultdict
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter
from insightface.utils import face_align

def cosine_similarity(emb1, emb2):
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            return np.dot(emb1, emb2)


def find_most_similar_face(main_pic_faces, target_face):
            
    similarities = [cosine_similarity(face.embedding, target_face.embedding) for face in main_pic_faces]
    best_match_idx = np.argmax(similarities)
    best_face = main_pic_faces[best_match_idx]
            
    return best_face

def find_most_similar_face_with_score(faces, reference_embedding):
    best_face = None
    best_score = -1

    for face in faces:
        similarity = np.dot(face.normed_embedding, reference_embedding)
        if similarity > best_score:
            best_score = similarity
            best_face = face

    return best_face, best_score

def _crop_face(img, bbox, margin=0.2):
    """
    Crop the face region from the image using the bounding box.
    Adds optional margin (percentage of bbox size) around the face.
    """
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Calculate margin
    bbox_w, bbox_h = x2 - x1, y2 - y1
    dx = int(bbox_w * margin)
    dy = int(bbox_h * margin)

    # Apply margin and clamp to image bounds
    x1 = max(x1 - dx, 0)
    y1 = max(y1 - dy, 0)
    x2 = min(x2 + dx, w)
    y2 = min(y2 + dy, h)

    return img[y1:y2, x1:x2]

def extract_faces_from_img(app, image_path):

    image = cv2.imread(image_path)

    faces = app.get(image)

    face_entries = []

    for face in faces:
        crop = _crop_face(image, face.bbox)
        face_entries.append({
            "embedding": face.embedding,
            "face_crop": crop,
            "bbox": face.bbox
        })
    
    if len(face_entries) > 1:
        embeddings = np.array([entry["embedding"] for entry in face_entries])
        clustering = DBSCAN(eps=0.45, min_samples=1, metric='cosine').fit(embeddings)
        
        for entry, label in zip(face_entries, clustering.labels_):
            entry["cluster_id"] = label
    else:
        face_entries[0]["cluster_id"] = 0

    return face_entries

def cluster_faces_from_video(video_path, app, sample_every_n=5,
                             eps=0.7, min_samples=8, merge_thresh=0.35):
    """
    Processes a video, extracts face embeddings, clusters them, merges similar clusters,
    and returns thumbnails with cluster IDs.

    Parameters:
    - video_path: str
    - app: initialized insightface.FaceAnalysis
    - sample_every_n: int (how frequently to sample frames)
    - eps: float (DBSCAN epsilon)
    - min_samples: int (DBSCAN min_samples)
    - merge_thresh: float (cosine threshold for post-merge)

    Returns:
    - face_db: list of all face records with cluster_id assigned
    - cluster_samples: list of (cluster_id, face crop) pairs for visualization
    """
    face_db = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            faces = app.get(frame)
            for face in faces:
                cropped = _crop_face(frame, face.bbox)
                face_db.append({
                    "frame_index": frame_idx,
                    "bbox": face.bbox,
                    "embedding": face.embedding,
                    "landmarks": face.kps,
                    "crop": cropped
                })

        frame_idx += 1

    cap.release()
    print(f"✅ Total number of faces detected: {len(face_db)}")

    # Normalize embeddings
    embeddings = np.array([entry["embedding"] for entry in face_db])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # DBSCAN clustering
    clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    for entry, label in zip(face_db, clusterer.labels_):
        entry["cluster_id"] = label

    # Post-processing: Merge similar clusters
    cluster_faces = defaultdict(list)
    cluster_means = {}

    for entry in face_db:
        cid = entry["cluster_id"]
        if cid == -1:
            continue
        cluster_faces[cid].append(entry["embedding"])

    for cid, embs in cluster_faces.items():
        cluster_means[cid] = np.mean(embs, axis=0)

    merge_map = {}
    cluster_ids = list(cluster_means.keys())
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cid1, cid2 = cluster_ids[i], cluster_ids[j]
            dist = cosine(cluster_means[cid1], cluster_means[cid2])
            if dist < merge_thresh:
                merge_map[cid2] = cid1

    for entry in face_db:
        cid = entry["cluster_id"]
        while cid in merge_map:
            cid = merge_map[cid]
        entry["cluster_id"] = cid

    # Cluster thumbnail collection
    seen = set()
    cluster_samples = []
    for entry in face_db:
        cid = entry["cluster_id"]
        if cid == -1 or cid in seen:
            continue
        seen.add(cid)
        cluster_samples.append({
            "cluster_id": cid,
            "face_crop": entry["crop"]
        })

    return face_db, cluster_samples


def get_mean_embedding_from_cluster(face_db, chosen_cluster_id):
    """
    Given a face_db and a cluster ID, return the mean embedding of that cluster.

    Parameters:
    - face_db: list of face entries, each with 'embedding' and 'cluster_id'
    - chosen_cluster_id: int, the ID of the cluster to extract mean from

    Returns:
    - mean_embedding: np.ndarray of shape (512,) or None if cluster is invalid
    """
    cluster_embeddings = [
        entry["embedding"] for entry in face_db
        if entry["cluster_id"] == chosen_cluster_id
    ]

    if not cluster_embeddings:
        print(f"❌ No faces found in cluster {chosen_cluster_id}")
        return None

    mean_embedding = np.mean(cluster_embeddings, axis=0)
    mean_embedding /= np.linalg.norm(mean_embedding)
    return mean_embedding


def high_res_image_to_unicode(frame, new_width=80):
    img = Image.fromarray(frame).convert("RGB")
    terminal_width, terminal_height = shutil.get_terminal_size()

    width, height = img.size
    aspect_ratio = height / width
    if new_width is None:
        new_width = terminal_width
    
    new_height = int(new_width * aspect_ratio * 0.55) * 2
    if new_height > terminal_height * 2:  # Double height since doubling resolution
        new_height = (terminal_height - 2) * 2
        new_width = int(new_height / (aspect_ratio * 0.55 * 2))
    resized_img = img.resize((new_width, new_height))
    
    result = []
    for y in range(0, new_height - 1, 2):
        for x in range(new_width):
            
            top_pixel = resized_img.getpixel((x, y))
            bottom_pixel = resized_img.getpixel((x, y + 1))
            
            # Create half blocks with different colors for top and bottom
            result.append(_high_res_rgb_pixels(top_pixel, bottom_pixel))
        result.append('\n')
    
    return ''.join(result)

def _high_res_rgb_pixels(upper_pixel, lower_pixel):
    return f"\033[38;2;{upper_pixel[0]};{upper_pixel[1]};{upper_pixel[2]}m\033[48;2;{lower_pixel[0]};{lower_pixel[1]};{lower_pixel[2]}m▀\033[0m"


# def display_faces_terminal(face_entries, max_faces=5, resize=(32, 32)):
#     """
#     Displays cropped face images in the terminal using rich_pixels.

#     Parameters:
#     - face_entries: List of dicts, each with keys "face_crop" (BGR image) and "cluster_id"
#     - max_faces: Max number of faces to display
#     - resize: Tuple (width, height) to resize face crops for terminal output
#     """
#     console = Console()
#     panels = []
    
#     for i, entry in enumerate(face_entries[:max_faces]):
#         bgr_crop = entry["face_crop"]
#         cluster_id = entry.get("cluster_id", "?")
        
#         # Convert BGR → RGB → PIL
#         rgb_img = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(rgb_img).resize(resize)
        
#         # Wrap in panel with label
#         panel = Panel(
#             Pixels.from_image(pil_img),
#             title=f"[bold]#{i}[/] C{cluster_id}",
#             padding=0,
#             expand=False
#         )
#         panels.append(panel)
#     console.print(Columns(panels, expand=False, align="center"))

def display_faces_terminal(face_entries, max_faces=5, resize_width=20):
    """
    Displays cropped face images in the terminal using high_res_image_to_unicode.

    Parameters:
    - face_entries: List of dicts, each with keys "face_crop" (BGR image) and "cluster_id"
    - max_faces: Max number of faces to display
    - resize_width: Width for resizing the image for terminal display
    """
    for i, entry in enumerate(face_entries[:max_faces]):
        bgr_crop = entry["face_crop"]
        cluster_id = entry.get("cluster_id", "?")

        # Convert BGR to RGB (OpenCV to PIL format)
        rgb_img = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

        # Print label
        print(f"\n🧠 Face {i} | Cluster {cluster_id}\n")

        # Convert and render the image as unicode block
        print(high_res_image_to_unicode(rgb_img, new_width=resize_width))

def _get_rotation_from_face(img, face):
    box = face.bbox.astype(int)
    w, h = box[2] - box[0], box[3] - box[1]
    kps = face.kps  # [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    if w > h:
        # sideways
        left_eye, right_eye = kps[0], kps[1]
        if right_eye[1] > left_eye[1]:
            aligned_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return aligned_img, cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            aligned_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return aligned_img, cv2.ROTATE_90_CLOCKWISE
    else:
        # upright-ish
        nose = kps[2]
        eye_y = (kps[0][1] + kps[1][1]) / 2
        if eye_y > nose[1]:
            # eyes below nose, face is upside down
            aligned_img = cv2.rotate(img, cv2.ROTATE_180)
            return aligned_img, cv2.ROTATE_180
        else:
            # upright
            return img, None

def _transform_bbox_after_rotation(bbox, original_shape, rotation_type):
    if rotation_type is None:
        return bbox
        
    orig_h, orig_w = original_shape[:2]
    x1, y1, x2, y2 = bbox
    
    if rotation_type == cv2.ROTATE_90_CLOCKWISE:
        new_x1 = orig_h - 1 - y2
        new_y1 = x1
        new_x2 = orig_h - 1 - y1
        new_y2 = x2
    elif rotation_type == cv2.ROTATE_90_COUNTERCLOCKWISE:
        new_x1 = y1
        new_y1 = orig_w - 1 - x2
        new_x2 = y2
        new_y2 = orig_w - 1 - x1
    elif rotation_type == cv2.ROTATE_180:
        new_x1 = orig_w - 1 - x2
        new_y1 = orig_h - 1 - y2
        new_x2 = orig_w - 1 - x1
        new_y2 = orig_h - 1 - y1
    else:
        return bbox
    
    return np.array([new_x1, new_y1, new_x2, new_y2])

def _compute_iou(boxA, boxB):
    """Calculate Intersection over Union of two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # no intersection
    if xB <= xA or yB <= yA:
        return 0.0
    
    interArea = (xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

def _rotate_back(img, rotation_type):
    if rotation_type is None:
        return img
    elif rotation_type == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_type == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_type == cv2.ROTATE_180:
        return cv2.rotate(img, cv2.ROTATE_180)
    else:
        return img

def aligned_highres_swap(swapper, app, img, target_face, source_face, upscale=1, restore_mouth=False):
    """
    Robust high-res face swap with automatic tilt correction and multi-face support.
    
    Args:
        swapper: Face swapper model  
        app: Face detector/analyzer
        img: Input image
        target_face: Face object to be replaced
        source_face: Face object to use as replacement
        upscale: Upscale factor for high-res processing
        restore_mouth: Whether to restore mouth details
        
    Returns:
        Swapped image, or original image if swap fails
    """
    try:
        original_shape = img.shape
        
        # rotate image to make face upright
        aligned_img, rotation_type = _get_rotation_from_face(img, target_face)
        
        aligned_faces = app.get(aligned_img)
        
        if not aligned_faces:
            print("No faces detected in aligned image, falling back to original")
            return highres_swap(swapper, img, target_face, source_face, upscale=upscale, restore_mouth=restore_mouth)
        
        # transform original bbox to aligned image coordinates
        expected_bbox = _transform_bbox_after_rotation(
            target_face.bbox, original_shape, rotation_type
        )
        
        # iou filtering for face
        best_face = None
        best_iou = 0.0
        
        for face in aligned_faces:
            iou = _compute_iou(expected_bbox, face.bbox)
            if iou > best_iou:
                best_iou = iou
                best_face = face
        
        # fallback to largest face
        if best_face is None or best_iou < 0.1:
            print(f"Poor face match (IoU: {best_iou:.3f}), using largest face")
            # Sort faces by area and pick the largest
            best_face = max(aligned_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # high-res swap on aligned image
        swapped_aligned = highres_swap(
            swapper, aligned_img, best_face, source_face, 
            upscale=upscale, restore_mouth=restore_mouth
        )
        
        # realign back
        final_result = _rotate_back(swapped_aligned, rotation_type)
        
        return final_result
        
    except Exception as e:
        print(f"aligned_highres_swap Error")
        try:
            return highres_swap(swapper, img, target_face, source_face, upscale=upscale, restore_mouth=restore_mouth)
        except:
            return img

def highres_swap(inswapper, img, target_face, source_face, upscale=1, restore_mouth=False):
    """
    Performs high-resolution face swapping using a tile-based "pixel-boost" pipeline.
    This mimics the approach used by roop-floyd and RopePearl to overcome the
    128x128 limitation of the InSwapper model.

    Args:
        inswapper: An instance of the INSwapper model.
        img: The original image (NumPy array, BGR format).
        target_face: A Face object representing the target face in `img`.
        source_face: A Face object representing the source face to swap from.
        upscale: The scaling factor for the output resolution (e.g., 2 for 2x, 4 for 4x).
                 Output resolution will be (128 * upscale) x (128 * upscale).
        restore_mouth: A Boolean to decide whether the original mouth be masked onto the output image.

    Returns:
        The image with the high-resolution swapped face (NumPy array, BGR format).
    """

    model_output_size = inswapper.input_size[0] # This is typically 128
    subsample_size = model_output_size * upscale # Desired output size (e.g., 256 or 512)
    pixel_boost_total = upscale # Number of tiles along one dimension (e.g., 2 for 2x2, 4 for 4x4)

    def _implode_pixel_boost(aligned_face_frame, model_size, pixel_boost_total):
        """
        Splits a large aligned face frame into smaller tiles.
        (S, S, 3) -> (N^2, M, M, 3) where S = N*M
        """
        subsample_frame = aligned_face_frame.reshape(model_size, pixel_boost_total, model_size, pixel_boost_total, 3)
        subsample_frame = subsample_frame.transpose(1, 3, 0, 2, 4).reshape(pixel_boost_total ** 2, model_size, model_size, 3)
        return subsample_frame

    def _explode_pixel_boost(subsample_frames, model_size, pixel_boost_total, pixel_boost_size):
        """
        Reconstructs a large swapped face frame from smaller tiles.
        (N^2, M, M, 3) -> (S, S, 3)
        """
        final_frame = np.stack(subsample_frames, axis=0).reshape(pixel_boost_total, pixel_boost_total, model_size, model_size, 3)
        final_frame = final_frame.transpose(2, 0, 3, 1, 4).reshape(pixel_boost_size, pixel_boost_size, 3)
        return final_frame

    def _prepare_crop_frame(swap_frame):
        """
        Pre-processes a tile for direct ONNX model input.
        (H, W, C) BGR -> (1, C, H, W) RGB, normalized float32
        """
        # BGR to RGB, normalize to [0, 1]
        swap_frame = swap_frame[:, :, ::-1] / 255.0
        # Transpose HWC to CHW
        swap_frame = swap_frame.transpose(2, 0, 1)
        # Add batch dimension and convert to float32
        swap_frame = np.expand_dims(swap_frame, axis=0).astype(np.float32)
        return swap_frame

    def _normalize_swap_frame(swap_frame_output):
        """
        Post-processes the ONNX model output back to image format.
        (1, C, H, W) float32 -> (H, W, C) BGR uint8
        """
        # Remove batch dimension
        swap_frame_output = swap_frame_output[0]
        # Transpose CHW to HWC
        swap_frame_output = swap_frame_output.transpose(1, 2, 0)
        # Denormalize and convert to uint8
        swap_frame_output = (swap_frame_output * 255.0).round()
        # RGB to BGR
        swap_frame_output = swap_frame_output[:, :, ::-1]
        return swap_frame_output.astype(np.uint8)

    def _create_mouth_mask(face, frame):
        mouth_cutout = None
        
        landmarks = face.landmark_2d_106
        if landmarks is not None:
            # Get mouth landmarks (indices 52 to 71 typically represent the outer mouth)
            mouth_points = landmarks[52:71].astype(np.int32)
            
            # Add padding to mouth area
            min_x, min_y = np.min(mouth_points, axis=0)
            max_x, max_y = np.max(mouth_points, axis=0)
            min_x = max(0, min_x - (15*6))
            min_y = max(0, min_y - 22)
            max_x = min(frame.shape[1], max_x + (15*6))
            max_y = min(frame.shape[0], max_y + (90*6))
            
            # Extract the mouth area from the frame using the calculated bounding box
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        return mouth_cutout, (min_x, min_y, max_x, max_y)

    def _create_feathered_mask(shape, feather_amount=30):
        mask = np.zeros(shape[:2], dtype=np.float32)
        center = (shape[1] // 2, shape[0] // 2)
        cv2.ellipse(mask, center, (shape[1] // 2 - feather_amount, shape[0] // 2 - feather_amount), 
                    0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
        return mask / np.max(mask)

    def _apply_color_transfer(source, target):
        """
        Apply color transfer from target to source image
        """
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        source_mean, source_std = cv2.meanStdDev(source)
        target_mean, target_std = cv2.meanStdDev(target)

        # Reshape mean and std to be broadcastable
        source_mean = source_mean.reshape(1, 1, 3)
        source_std = source_std.reshape(1, 1, 3)
        target_mean = target_mean.reshape(1, 1, 3)
        target_std = target_std.reshape(1, 1, 3)

        # Perform the color transfer
        source = (source - source_mean) * (target_std / source_std) + target_mean
        return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)


    def _apply_mouth_area(frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple) -> np.ndarray:
        min_x, min_y, max_x, max_y = mouth_box
        box_width = max_x - min_x
        box_height = max_y - min_y        

        # Resize the mouth cutout to match the mouth box size
        if mouth_cutout is None or box_width is None or box_height is None:
            return frame
        try:
            resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
            
            # Extract the region of interest (ROI) from the target frame
            roi = frame[min_y:max_y, min_x:max_x]
            
            # Ensure the ROI and resized_mouth_cutout have the same shape
            if roi.shape != resized_mouth_cutout.shape:
                resized_mouth_cutout = cv2.resize(resized_mouth_cutout, (roi.shape[1], roi.shape[0]))
            
            # Apply color transfer from ROI to mouth cutout
            color_corrected_mouth = _apply_color_transfer(resized_mouth_cutout, roi)
            
            # Create a feathered mask with increased feather amount
            feather_amount = min(30, box_width // 15, box_height // 15)
            mask = _create_feathered_mask(resized_mouth_cutout.shape, feather_amount)
            
            # Blend the color-corrected mouth cutout with the ROI using the feathered mask
            mask = mask[:,:,np.newaxis]  # Add channel dimension to mask
            blended = (color_corrected_mouth * mask + roi * (1 - mask)).astype(np.uint8)
            
            # Place the blended result back into the frame
            frame[min_y:max_y, min_x:max_x] = blended
        except Exception as e:
            print(f'Error {e}')
            pass

        return frame

    # Step 1: Align and upscale the target face to the desired high resolution
    aligned_face, M_affine = face_align.norm_crop2(img, target_face.kps, subsample_size)

    # Step 2: Implode the aligned high-res face into 128x128 tiles
    subsample_frames = _implode_pixel_boost(aligned_face, model_output_size, pixel_boost_total)

    # Step 3: Prepare latent embedding for the source face (done once)
    latent = source_face.normed_embedding.reshape(1, -1)
    latent = np.dot(latent, inswapper.emap)
    latent /= np.linalg.norm(latent)

    # Step 4: Swap each tile using the direct ONNX session call
    swapped_tiles = []
    for sliced_frame in subsample_frames:
        # Pre-process the tile
        prepared_tile_blob = _prepare_crop_frame(sliced_frame)

        # Run the ONNX model session on the prepared tile and latent embedding
        pred = inswapper.session.run(inswapper.output_names, {
            inswapper.input_names[0]: prepared_tile_blob,
            inswapper.input_names[1]: latent
        })[0]

        # Post-process the swapped tile
        normalized_swapped_tile = _normalize_swap_frame(pred)
        swapped_tiles.append(normalized_swapped_tile)

    # Step 5: Reconstruct the full high-res face image from swapped tiles
    highres_face = _explode_pixel_boost(swapped_tiles, model_output_size, pixel_boost_total, subsample_size)

    # Step 6: Paste the high-res swapped face back into the original image
    IM = cv2.invertAffineTransform(M_affine)
    
    # Create initial white mask
    img_white = np.full((subsample_size, subsample_size), 255, dtype=np.float32)
    
    # Warp both the face and mask back to original image coordinates
    face_img = cv2.warpAffine(highres_face, IM, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    img_mask = cv2.warpAffine(img_white, IM, (img.shape[1], img.shape[0]), borderValue=0.0)
    
    # Clean up the mask
    img_mask[img_mask > 20] = 255
    
    # Calculate mask dimensions for adaptive kernel size
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        
        # Erode the mask
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        
        # Apply Gaussian blur
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    
    # Normalize mask
    img_mask = img_mask.astype(np.float32) / 255.0
    
    # Expand mask to 3 channels
    img_mask_3ch = np.stack([img_mask] * 3, axis=-1)
    
    # Blend the images
    out_img = img_mask_3ch * face_img.astype(np.float32) + (1 - img_mask_3ch) * img.astype(np.float32)
    out_img = out_img.astype(np.uint8)

    if restore_mouth:
        mouth_cutout, mouth_bbox = _create_mouth_mask(target_face, img)
        out_img = _apply_mouth_area(frame=out_img, mouth_cutout=mouth_cutout, mouth_box=mouth_bbox)

    return out_img