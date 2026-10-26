import json
import ollama
from database import SessionLocal
from models import TranscriptSegment, Caption, Video

BATCH_SIZE = 12  # fewer total calls than before, still small enough for reliable JSON output

LANGUAGE_NAMES = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
}


def _translate_batch(texts, language_name):
    """Translate a list of texts in one call. Returns a list of the same
    length on success, or None if the model's output couldn't be trusted."""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    prompt = f"""Translate each numbered line below into {language_name}.

STRICT RULES:
- The entire translation must be written ONLY in {language_name} script and vocabulary.
- Do NOT leave any words untranslated, even technical or academic terms — translate them into their standard {language_name} equivalent.
- Do NOT insert any words or characters from any other language.
- Numbers, mathematical symbols (like n, x, +, =), and proper nouns may stay as-is.
- Keep the meaning and tone natural, not overly literal.

{numbered}

Respond with ONLY a JSON array of strings, in the same order, with exactly {len(texts)} items — no other text, no markdown formatting.
Example format: ["translation 1", "translation 2"]"""

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list) and len(result) == len(texts):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    return None


def _translate_single(text, language_name):
    """Fallback: translate one segment at a time if batch translation fails."""
    prompt = f"""Translate this into {language_name}. Write the ENTIRE translation only in {language_name} — do not leave any words in English or any other language, including technical terms. Respond with ONLY the translation, no other text:

{text}"""
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


def generate_captions_for_language(video_id, language_code):
    """Generates (or regenerates) captions for ONE language only.
    Leaves any other already-generated languages untouched."""
    db = SessionLocal()

    if language_code not in LANGUAGE_NAMES:
        db.close()
        return None

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.start_time)
        .all()
    )

    if not segments:
        db.close()
        return None

    # Figure out what language the video was actually spoken/transcribed in.
    # This matters because "en" captions used to always mean "just copy the
    # transcript" — which was wrong for a non-English video, since the
    # transcript itself isn't English in that case.
    video = db.query(Video).filter(Video.id == video_id).first()
    source_language = video.detected_language if video and video.detected_language else "en"

    db.query(Caption).filter(
        Caption.video_id == video_id,
        Caption.language == language_code
    ).delete()
    db.commit()

    count = 0

    if language_code == "en" and source_language == "en":
        # Source is already English — safe to just copy, no translation needed
        for seg in segments:
            db.add(Caption(
                video_id=video_id,
                language="en",
                start_time=seg.start_time,
                end_time=seg.end_time,
                content=seg.content
            ))
            count += 1
        db.commit()
    else:
        # Either translating INTO English from a non-English source,
        # or translating into one of the other target languages
        language_name = "English" if language_code == "en" else LANGUAGE_NAMES[language_code]

        for i in range(0, len(segments), BATCH_SIZE):
            batch = segments[i:i + BATCH_SIZE]
            texts = [s.content for s in batch]

            translations = _translate_batch(texts, language_name)

            if translations is None:
                translations = [_translate_single(t, language_name) for t in texts]

            for seg, translated_text in zip(batch, translations):
                db.add(Caption(
                    video_id=video_id,
                    language=language_code,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    content=translated_text
                ))
                count += 1

            db.commit()

    db.close()
    return {"language": language_code, "captions_created": count}


if __name__ == "__main__":
    video_id = 14
    result = generate_captions_for_language(video_id, "ar")
    print(result)
