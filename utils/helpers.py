import numpy as np

def cosine_similarity(emb1, emb2):
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            return np.dot(emb1, emb2)


def find_most_similar_face(main_pic_faces, target_face):
            
    similarities = [cosine_similarity(face.embedding, target_face.embedding) for face in main_pic_faces]
    best_match_idx = np.argmax(similarities)
    best_face = main_pic_faces[best_match_idx]
            
    return best_face

def find_most_similar_face_with_score(faces, reference_face):
    best_face = None
    best_score = -1

    for face in faces:
        similarity = np.dot(face.normed_embedding, reference_face.normed_embedding)
        if similarity > best_score:
            best_score = similarity
            best_face = face

    return best_face, best_score
