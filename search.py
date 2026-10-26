import ollama
import numpy as np
from models import TranscriptSegment, Video, User
from database import SessionLocal
from models import TranscriptSegment


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_segments(question, video_id=None, top_k=3):
    db = SessionLocal()

    response = ollama.embeddings(model="nomic-embed-text", prompt=question)
    question_embedding = response["embedding"]

    query = db.query(TranscriptSegment)
    if video_id:
        query = query.filter(TranscriptSegment.video_id == video_id)
    segments = query.all()

    scored = []
    for segment in segments:
        if not segment.embedding_vector:
            continue  # skip segments that were never embedded
        similarity = cosine_similarity(question_embedding, segment.embedding_vector)
        scored.append((similarity, segment))

    scored.sort(key=lambda x: x[0], reverse=True)

    db.close()
    return scored[:top_k]
def search_videos(query, top_k=5):
    """Searches across all instructor-uploaded, ready videos.
    Returns each matching video once, ranked by its single best-matching segment."""
    db = SessionLocal()

    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    query_embedding = response["embedding"]

    results = (
        db.query(TranscriptSegment, Video)
        .join(Video, TranscriptSegment.video_id == Video.id)
        .join(User, Video.owner_id == User.id)
        .filter(User.role == "instructor")
        .filter(Video.status == "ready")
        .all()
    )

    best_per_video = {}

    for segment, video in results:
        if not segment.embedding_vector:
            continue
        score = cosine_similarity(query_embedding, segment.embedding_vector)

        if video.id not in best_per_video or score > best_per_video[video.id]["score"]:
            best_per_video[video.id] = {
                "score": score,
                "video_id": video.id,
                "title": video.title,
                "snippet": segment.content,
                "start_time": segment.start_time,
            }

    ranked = sorted(best_per_video.values(), key=lambda x: x["score"], reverse=True)
    ranked = [r for r in ranked if r["score"] >= 0.54]  # drop weak/coincidental matches

    db.close()
    return ranked[:top_k]