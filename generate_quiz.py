import json
import ollama
from database import SessionLocal
from models import TranscriptSegment, Quiz, QuizQuestion, QuizOption, Video

CHUNK_SECONDS = 300
MIN_QUESTIONS = 4
MAX_QUESTIONS = 10


def _parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def generate_quiz_for_video(video_id):
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

    video = db.query(Video).filter(Video.id == video_id).first()
    quiz_title = f"Quiz — {video.title}" if video else f"Quiz for video {video_id}"

    total_duration = segments[-1].end_time
    num_questions = max(MIN_QUESTIONS, min(MAX_QUESTIONS, round(total_duration / CHUNK_SECONDS)))
    chunk_length = total_duration / num_questions

    old_quiz = db.query(Quiz).filter(Quiz.video_id == video_id).first()
    if old_quiz:
        old_question_ids = [q.id for q in db.query(QuizQuestion).filter(QuizQuestion.quiz_id == old_quiz.id).all()]
        for qid in old_question_ids:
            db.query(QuizOption).filter(QuizOption.question_id == qid).delete()
        db.query(QuizQuestion).filter(QuizQuestion.quiz_id == old_quiz.id).delete()
        db.query(Quiz).filter(Quiz.id == old_quiz.id).delete()
        db.commit()

    new_quiz = Quiz(video_id=video_id, title=quiz_title)
    db.add(new_quiz)
    db.commit()
    db.refresh(new_quiz)

    created_count = 0

    for i in range(num_questions):
        chunk_start = i * chunk_length
        chunk_end = (i + 1) * chunk_length

        chunk_segments = [s for s in segments if chunk_start <= s.start_time < chunk_end]
        if not chunk_segments:
            continue

        chunk_text = "\n".join(f"[{s.start_time:.1f}s] {s.content}" for s in chunk_segments)
        question_type = "mcq" if i % 2 == 0 else "written"

        if question_type == "mcq":
            prompt = f"""Based on this lecture transcript segment, write ONE multiple-choice question testing understanding of the main idea.

Transcript segment:
{chunk_text}

Write the question and all answer options in English only, even if the transcript segment above is in a different language. Do not use any non-English words or characters.

Respond with ONLY a JSON object, no other text, no markdown formatting:
{{"question": "...", "options": [{{"text": "...", "is_correct": true}}, {{"text": "...", "is_correct": false}}, {{"text": "...", "is_correct": false}}, {{"text": "...", "is_correct": false}}]}}

Exactly one option must have "is_correct": true. Respond with ONLY the JSON object."""
        else:
            prompt = f"""Based on this lecture transcript segment, write ONE short-answer question testing understanding of the main idea, plus a concise ideal/reference answer.

Transcript segment:
{chunk_text}

Write the question and the reference answer in English only, even if the transcript segment above is in a different language. Do not use any non-English words or characters.

Respond with ONLY a JSON object, no other text, no markdown formatting:
{{"question": "...", "reference_answer": "..."}}

Respond with ONLY the JSON object."""

        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = response["message"]["content"]

        try:
            data = _parse_json(raw)
        except json.JSONDecodeError:
            print(f"--- Could not parse quiz chunk {i} ({question_type}) ---")
            print(raw)
            continue

        if question_type == "mcq":
            correct_option = next((o["text"] for o in data["options"] if o.get("is_correct")), None)
            new_question = QuizQuestion(
                quiz_id=new_quiz.id,
                question_text=data["question"],
                question_type="mcq",
                correct_answer=correct_option
            )
            db.add(new_question)
            db.commit()
            db.refresh(new_question)

            for opt in data["options"]:
                db.add(QuizOption(
                    question_id=new_question.id,
                    option_text=opt["text"],
                    is_correct=bool(opt.get("is_correct"))
                ))
            db.commit()
        else:
            new_question = QuizQuestion(
                quiz_id=new_quiz.id,
                question_text=data["question"],
                question_type="written",
                correct_answer=data["reference_answer"]
            )
            db.add(new_question)
            db.commit()

        created_count += 1

    
    quiz_id = new_quiz.id
    db.close()
    return {"quiz_id": quiz_id, "questions_created": created_count}


if __name__ == "__main__":
    video_id = 10
    result = generate_quiz_for_video(video_id)
    print(result)