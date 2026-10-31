from faster_whisper import WhisperModel

#Load the speech recognition model
model = WhisperModel(
    "base",
    device ="cpu",
    compute_type="int8"
)

def transcribe_audio(audio_path: str) -> str:
    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    text = "".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()
