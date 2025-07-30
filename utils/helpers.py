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
