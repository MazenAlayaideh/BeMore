import ollama
from database import SessionLocal
from models import TranscriptSegment


def generate_embeddings_for_video(video_id):
    db = SessionLocal()

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.video_id == video_id)
        .filter(TranscriptSegment.embedding_vector.is_(None))
        .all()
    )

    for segment in segments:
        response = ollama.embeddings(model="nomic-embed-text", prompt=segment.content)
        segment.embedding_vector = response["embedding"]
        db.commit()

    count = len(segments)
    db.close()
    return count


if __name__ == "__main__":
    video_id = 12  # change to backfill a specific video
    count = generate_embeddings_for_video(video_id)
    print(f"Embedded {count} segments for video_id={video_id}")