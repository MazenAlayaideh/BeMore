import ollama
import numpy as np
from database import SessionLocal
from models import TranscriptSegment

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(question, video_id=None, top_k=3):
    db = SessionLocal()

    # Embed the question
    response = ollama.embeddings(model="nomic-embed-text", prompt=question)
    question_embedding = response["embedding"]

    # Get all segments (optionally filtered to one video)
    query = db.query(TranscriptSegment)
    if video_id:
        query = query.filter(TranscriptSegment.video_id == video_id)
    segments = query.all()

    # Score each segment by similarity to the question
    scored = []
    for segment in segments:
        similarity = cosine_similarity(question_embedding, segment.embedding_vector)
        scored.append((similarity, segment))

    # Sort by similarity, highest first
    scored.sort(key=lambda x: x[0], reverse=True)

    db.close()
    return scored[:top_k]


if __name__ == "__main__":
    question = "What is the difference between a nice guy and a good man?"
    results = search(question, video_id=3, top_k=3)

    print(f"\nQuestion: {question}\n")
    print("--- TOP MATCHING SEGMENTS ---")
    for similarity, segment in results:
        print(f"[similarity: {similarity:.3f}] [{segment.start_time:.1f}s - {segment.end_time:.1f}s] {segment.content}")