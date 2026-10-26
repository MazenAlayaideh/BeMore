import whisper

# Path to a video/audio file already in your uploads folder
file_path = file_path = "uploads/videoplayback.mp4"

print("Loading Whisper model... (this may take a minute the first time)")
model = whisper.load_model("base")

print("Transcribing... (this can take a while depending on video length)")
result = model.transcribe(file_path)

print("\n--- FULL TRANSCRIPT ---")
print(result["text"])

print("\n--- TIMESTAMPED SEGMENTS ---")
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.1f}s - {end:.1f}s] {text}")