import json
import ollama
from database import SessionLocal
from models import TranscriptSegment, Topic

CHUNK_SECONDS = 390  # ~6.5 minutes per chunk
MIN_TOPICS = 3
MAX_TOPICS = 14


def generate_topics_for_video(video_id):
    db = SessionLocal()

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.start_time)
        .all()
    )

    if not segments:
        db.close()
        return None

    db.query(Topic).filter(Topic.video_id == video_id).delete()
    db.commit()

    video_duration = segments[-1].end_time
    num_chunks = max(MIN_TOPICS, min(MAX_TOPICS, int(video_duration // CHUNK_SECONDS) + 1))
    chunk_length = video_duration / num_chunks

    previous_titles = []
    saved_topics = []

    for i in range(num_chunks):
        chunk_start = i * chunk_length
        chunk_end = chunk_start + chunk_length

        chunk_segments = [s for s in segments if chunk_start <= s.start_time < chunk_end]
        if not chunk_segments:
            continue

        chunk_text = " ".join(s.content for s in chunk_segments)

        previous_titles_note = ""
        if previous_titles:
            titles_list = ", ".join(f'"{t}"' for t in previous_titles[-3:])
            previous_titles_note = f"\nTopics already used earlier in this lecture: {titles_list}. Do NOT repeat these titles — if this chunk continues the same broad idea, pick a more specific sub-point instead."

        prompt = f"""This is one segment of a longer lecture transcript, covering roughly {chunk_start:.0f}s to {chunk_end:.0f}s.

Transcript segment:
{chunk_text}
{previous_titles_note}

Always write the "title" in English, even if the transcript segment above is in a different language. Do not use any non-English words or characters in the title.

Pick a "start_time" as EARLY as possible within this segment — specifically, the moment this chunk's main idea first begins, not a moment deep into the chunk. Prefer a start_time within the first half of this chunk's time range whenever the content allows it.

Respond with ONLY a JSON object, no other text, no markdown formatting, with exactly these fields:
- "title": a short, clear label (3-6 words) for the MAIN idea discussed in this segment
- "start_time": the timestamp (in seconds, as a number) where this main idea actually begins within this segment

Example format:
{{"title": "Proof by induction", "start_time": 122.5}}

Respond with ONLY the JSON object."""

        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = response["message"]["content"].strip()

        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
            title = str(parsed["title"]).strip()
            start_time = float(parsed["start_time"])
        except Exception:
            continue

        # Keep the timestamp sane and inside this chunk's own window
        start_time = max(chunk_start, min(start_time, chunk_end))

        saved_topics.append({"title": title, "start_time": start_time})
        previous_titles.append(title)

    saved_topics.sort(key=lambda t: t["start_time"])

    for t in saved_topics:
        db.add(Topic(video_id=video_id, title=t["title"], start_time=t["start_time"]))
    db.commit()
    db.close()

    return saved_topics


if __name__ == "__main__":
    video_id = 14  # change to a real video_id
    result = generate_topics_for_video(video_id)
    print(result)