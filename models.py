from sqlalchemy import Column, Integer, String, Float, Text, TIMESTAMP, JSON, Boolean
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(190), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="student")
    created_at = Column(TIMESTAMP, server_default=func.now())


class Video(Base):
    detected_language = Column(String(10), nullable=True)
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    filename = Column(String(255), nullable=False)
    duration_seconds = Column(Float, nullable=True)
    status = Column(String(20), default="uploaded")
    uploaded_at = Column(TIMESTAMP, server_default=func.now())


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    content = Column(Text, nullable=False)
    embedding_vector = Column(JSON, nullable=True)


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    start_time = Column(Float, nullable=False)


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    video_id = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(10), nullable=False)
    correct_answer = Column(Text, nullable=True)


class QuizOption(Base):
    __tablename__ = "quiz_options"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, nullable=False)
    option_text = Column(String(255), nullable=False)
    is_correct = Column(Boolean, default=False)


class Caption(Base):
    __tablename__ = "captions"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, nullable=False)
    language = Column(String(10), nullable=False)  # "en", "ar", "de", "es", "ja"
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    content = Column(Text, nullable=False)