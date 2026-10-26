import os
import shutil
import json as _json
import bcrypt
import ollama
import whisper
import threading
from typing import List
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import engine, get_db
from models import Video, TranscriptSegment, Topic, Note, Quiz, QuizQuestion, QuizOption, User, Caption
from search import search_segments
from generate_topics import generate_topics_for_video
from generate_quiz import generate_quiz_for_video
from generate_embeddings import generate_embeddings_for_video
from generate_captions import generate_captions_for_language
from search import search_segments, search_videos
from fastapi import FastAPI, UploadFile, File, Depends, Form

app = FastAPI(title="BeMore API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
CONFIDENCE_THRESHOLD = 0.55

os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

print("Loading Whisper model...")
whisper_model = whisper.load_model("small", device="cuda")
print("Whisper model loaded.")

transcription_lock = threading.Lock()
# --- Pydantic Models ---

class AskRequest(BaseModel):
    question: str
    video_id: int


class NoteCreate(BaseModel):
    video_id: int
    timestamp_seconds: float
    content: str
    user_id: int = 1


class QuizAnswer(BaseModel):
    question_id: int
    answer: str


class QuizSubmit(BaseModel):
    video_id: int
    answers: List[QuizAnswer]


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str  # "student" or "instructor"


class LoginRequest(BaseModel):
    email: str
    password: str


# --- Authentication Routes ---

@app.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        return {"status": "error", "message": "An account with this email already exists."}

    if req.role not in ("student", "instructor"):
        return {"status": "error", "message": "Role must be 'student' or 'instructor'."}

    hashed = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt())

    new_user = User(
        name=req.name,
        email=req.email,
        password_hash=hashed.decode("utf-8"),
        role=req.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "status": "success",
        "user": {"id": new_user.id, "name": new_user.name, "email": new_user.email, "role": new_user.role}
    }


@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        return {"status": "error", "message": "Incorrect email or password."}

    if not bcrypt.checkpw(req.password.encode("utf-8"), user.password_hash.encode("utf-8")):
        return {"status": "error", "message": "Incorrect email or password."}

    return {
        "status": "success",
        "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role}
    }


# --- Core Application Routes ---

@app.get("/")
def read_root():
    return {"message": "BeMore backend is running"}


@app.get("/test-db")
def test_db():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "success", "message": "Connected to MySQL successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/videos")
def list_videos(owner_id: int = None, db: Session = Depends(get_db)):
    if owner_id is not None:
        videos = db.query(Video).filter(Video.owner_id == owner_id).all()
    else:
        # General/public library view — only instructor-verified content,
        # matching the same role restriction already used for cross-video search
        videos = (
            db.query(Video)
            .join(User, Video.owner_id == User.id)
            .filter(User.role == "instructor")
            .all()
        )
    return [
        {
            "id": v.id,
            "title": v.title,
            "filename": v.filename,
            "status": v.status,
            "uploaded_at": v.uploaded_at.isoformat() if v.uploaded_at else None
        }
        for v in videos
    ]


@app.get("/videos/{video_id}")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        return {"error": "Video not found"}
    return {
        "id": video.id,
        "title": video.title,
        "filename": video.filename,
        "status": video.status,
        "uploaded_at": video.uploaded_at.isoformat() if video.uploaded_at else None
    }


@app.get("/topics/{video_id}")
def get_topics(video_id: int, db: Session = Depends(get_db)):
    topics = (
        db.query(Topic)
        .filter(Topic.video_id == video_id)
        .order_by(Topic.start_time)
        .all()
    )
    return [{"id": t.id, "title": t.title, "start_time": t.start_time} for t in topics]


@app.get("/notes/{video_id}")
def get_notes(video_id: int, user_id: int = 1, db: Session = Depends(get_db)):
    notes = (
        db.query(Note)
        .filter(Note.video_id == video_id, Note.user_id == user_id)
        .order_by(Note.timestamp_seconds)
        .all()
    )
    return [
        {
            "id": n.id,
            "timestamp_seconds": n.timestamp_seconds,
            "content": n.content,
            "created_at": n.created_at.isoformat() if n.created_at else None
        }
        for n in notes
    ]


@app.post("/notes")
def create_note(note: NoteCreate, db: Session = Depends(get_db)):
    new_note = Note(
        user_id=note.user_id,
        video_id=note.video_id,
        timestamp_seconds=note.timestamp_seconds,
        content=note.content
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)
    return {
        "id": new_note.id,
        "timestamp_seconds": new_note.timestamp_seconds,
        "content": new_note.content
    }

@app.delete("/notes/{note_id}")
def delete_note(note_id: int, db: Session = Depends(get_db)):
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        return {"status": "error", "message": "Note not found."}
    db.delete(note)
    db.commit()
    return {"status": "success"}

@app.post("/quiz/generate/{video_id}")
def generate_quiz(video_id: int):
    result = generate_quiz_for_video(video_id)
    if result is None:
        return {"status": "error", "message": "No transcript data for this video yet."}
    return {"status": "success", **result}


@app.get("/quiz/{video_id}")
def get_quiz(video_id: int, db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter(Quiz.video_id == video_id).order_by(Quiz.id.desc()).first()
    if not quiz:
        return {"quiz_id": None, "questions": []}

    questions = db.query(QuizQuestion).filter(QuizQuestion.quiz_id == quiz.id).all()
    output = []
    for q in questions:
        item = {"id": q.id, "question_text": q.question_text, "question_type": q.question_type}
        if q.question_type == "mcq":
            options = db.query(QuizOption).filter(QuizOption.question_id == q.id).all()
            item["options"] = [{"id": o.id, "option_text": o.option_text} for o in options]
        output.append(item)

    return {"quiz_id": quiz.id, "questions": output}


@app.post("/quiz/submit")
def submit_quiz(submission: QuizSubmit, db: Session = Depends(get_db)):
    results = []
    correct_count = 0

    for ans in submission.answers:
        question = db.query(QuizQuestion).filter(QuizQuestion.id == ans.question_id).first()
        if not question:
            continue

        if question.question_type == "mcq":
            chosen_option = db.query(QuizOption).filter(QuizOption.id == int(ans.answer)).first()
            is_correct = bool(chosen_option and chosen_option.is_correct)
            results.append({
                "question_id": question.id,
                "correct": is_correct,
                "correct_answer": question.correct_answer
            })
        else:
            student_answer = ans.answer.strip()

            if len(student_answer) < 10:
                results.append({
                    "question_id": question.id,
                    "correct": False,
                    "feedback": "Answer is too short to meaningfully address the question.",
                    "reference_answer": question.correct_answer
                })
                continue

            prompt = f"""You are strictly grading a student's short-answer response. Be a demanding grader — do not give credit for vague, off-topic, or incomplete answers.

Question: "{question.question_text}"
Reference/ideal answer: "{question.correct_answer}"
Student's answer: "{student_answer}"

Mark the answer correct ONLY if it demonstrates real understanding of the specific concept in the reference answer, even if worded differently. Mark it incorrect if it is vague, off-topic, too brief to show understanding, or simply repeats the question without explaining it.

Respond with ONLY a JSON object, no other text:
{{"correct": true or false, "feedback": "one short sentence explaining why"}}"""

            response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
            raw = response["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                graded = _json.loads(raw)
                is_correct = bool(graded.get("correct"))
                feedback = graded.get("feedback", "")
            except Exception:
                is_correct = False
                feedback = "Could not grade this answer automatically."

            results.append({
                "question_id": question.id,
                "correct": is_correct,
                "feedback": feedback,
                "reference_answer": question.correct_answer
            })

        if results[-1]["correct"]:
            correct_count += 1

    return {
        "total": len(results),
        "correct": correct_count,
        "results": results
    }


@app.post("/captions/generate/{video_id}")
def generate_captions(video_id: int, language: str = "en"):
    result = generate_captions_for_language(video_id, language)
    if result is None:
        return {"status": "error", "message": "No transcript data for this video, or invalid language code."}
    return {"status": "success", **result}


@app.get("/captions/{video_id}")
def get_captions(video_id: int, db: Session = Depends(get_db)):
    captions = (
        db.query(Caption)
        .filter(Caption.video_id == video_id)
        .order_by(Caption.language, Caption.start_time)
        .all()
    )
    return [
        {
            "language": c.language,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "content": c.content
        }
        for c in captions
    ]


@app.post("/upload")
def upload_video(file: UploadFile = File(...), owner_id: int = Form(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_video = Video(
        owner_id=owner_id,
        title=file.filename,
        filename=file.filename,
        status="uploaded"
    )
    db.add(new_video)
    db.commit()
    db.refresh(new_video)

    new_video.status = "processing"
    db.commit()

    with transcription_lock:
        result = whisper_model.transcribe(file_path)
    new_video.detected_language = result.get("language", "en")
    db.commit()
    
    for segment in result["segments"]:
        new_segment = TranscriptSegment(
            video_id=new_video.id,
            start_time=segment["start"],
            end_time=segment["end"],
            content=segment["text"].strip()
        )
        db.add(new_segment)

    db.commit()

    generate_embeddings_for_video(new_video.id)
    generate_topics_for_video(new_video.id)

    new_video.status = "ready"
    db.commit()

    return {
        "status": "success",
        "video_id": new_video.id,
        "filename": new_video.filename,
        "segments_created": len(result["segments"])
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    top_matches = search_segments(request.question, video_id=request.video_id, top_k=3)

    if not top_matches:
        return {
            "answer": "This video has no transcript data yet.",
            "source": "none",
            "citations": []
        }

    best_score = top_matches[0][0]

    if best_score >= CONFIDENCE_THRESHOLD:
        context_text = "\n".join(
            f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s] {seg.content}"
            for _, seg in top_matches
        )

        prompt = f"""You are a helpful study assistant. Answer the student's question using ONLY the lecture transcript excerpts below. Be concise and clear.

Transcript excerpts:
{context_text}

Question: {request.question}

Answer:"""

        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        answer_text = response["message"]["content"]

        # Safety check: even if similarity crossed the threshold, the model's own
        # answer might honestly report the transcript didn't actually cover this.
        # If so, correct the source tag instead of falsely claiming lecture-grounding.
        no_info_phrases = ["don't provide information", "doesn't provide information",
                            "no information on", "not mentioned in the transcript",
                            "no connection is made", "doesn't mention"]
        actually_grounded = not any(phrase in answer_text.lower() for phrase in no_info_phrases)

        return {
            "answer": answer_text,
            "source": "lecture_transcript" if actually_grounded else "general_knowledge",
            "citations": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.content,
                    "similarity": round(float(score), 3)
                }
                for score, seg in top_matches
            ] if actually_grounded else []
        }
    else:
        prompt = f"""Answer this question clearly and concisely, in the same language it was asked in. Stay strictly on topic — do not discuss anything unrelated to the question. Be clear that this is not from the student's lecture.

Question: {request.question}

Answer:"""

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )

        return {
            "answer": response["message"]["content"],
            "source": "general_knowledge",
            "citations": []
        }
@app.get("/search")
def search(q: str, limit: int = 5):
    results = search_videos(q, top_k=limit)
    return {"query": q, "results": results}

app.mount("/", StaticFiles(directory="public", html=True), name="frontend")
