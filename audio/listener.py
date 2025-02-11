from typing import Any, Dict, List

import librosa
import torch
import whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = whisper.load_model("base").to(device)


def get_audio_data(file_path: str) -> dict:
    """
    Loads an audio file using librosa.

    :param file_path: Path to the audio file.
    :return: Dictionary with keys 'data' and 'sr'.
    """
    data, sr = librosa.load(file_path, sr=None, mono=True)
    return {"data": data, "sr": sr}


def transcribe_audio_full(audio_data: dict, model_name: str = "base") -> List[Dict]:
    """
    Transcribes the entire audio using the global Whisper model.

    :param audio_data: Dictionary with 'data' and 'sr'.
    :param model_name: Model name (ignored because the global model is used).
    :return: List of transcription segments with 'start', 'end', and 'text'.
    """
    sr = audio_data["sr"]
    y = audio_data["data"]
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    result = MODEL.transcribe(y, fp16=True)
    return result.get("segments", [])  # type: ignore


def segments_to_phrases(segments: List[Dict]) -> List[Dict]:
    """
    Groups transcription segments into phrases. A new phrase is started when the text begins with an uppercase letter.

    :param segments: List of transcription segments.
    :return: List of phrases with 'start', 'end', and 'text'.
    """
    phrases = []
    current_phrase = None
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if text[0].isupper() or current_phrase is None:
            if current_phrase is not None:
                phrases.append(current_phrase)
            current_phrase = {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text,
            }
        else:
            if current_phrase is not None:
                current_phrase["text"] += " " + text
                current_phrase["end"] = seg.get("end", current_phrase["end"])
            else:
                current_phrase = {
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": text,
                }
    if current_phrase is not None:
        phrases.append(current_phrase)
    return phrases
