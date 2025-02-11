from typing import Any

import librosa
import numpy as np
import torch
import whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE: dict[str, Any] = {}


def get_model(model_name: str) -> Any:
    """
    Retrieves the Whisper model specified by model_name from the cache.
    If the model is not already loaded, it loads the model using whisper.load_model,
    moves it to the appropriate device (CUDA if available), caches it, and then returns it.

    :param model_name: The name of the Whisper model to load (e.g., "base", "large").
    :return: The loaded Whisper model.
    """
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = whisper.load_model(model_name).to(device)
    return MODEL_CACHE[model_name]


def get_audio_data(file_path: str) -> dict:
    """
    Loads an audio file using librosa.

    :param file_path: Path to the audio file.
    :return: Dictionary with keys 'data' and 'sr'.
    """
    data, sr = librosa.load(file_path, sr=None, mono=True)
    return {"data": data, "sr": sr}


def transcribe_audio_full(audio_data: dict, model_name: str = "base") -> list[dict]:
    """
    Transcribes the entire audio using the specified Whisper model.

    :param audio_data: Dictionary with 'data' and 'sr'.
    :param model_name: Model name to use for transcription.
    :return: List of transcription segments with 'start', 'end', and 'text'.
    """
    sr = audio_data["sr"]
    y = audio_data["data"]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    model = get_model(model_name)
    with torch.no_grad():
        result = model.transcribe(y, fp16=True)
    return result.get("segments", [])


def segments_to_phrases(segments: list[dict]) -> list[dict]:
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
            current_phrase["text"] += f" {text}"
            current_phrase["end"] = seg.get("end", current_phrase["end"])
    if current_phrase is not None:
        phrases.append(current_phrase)
    return phrases
