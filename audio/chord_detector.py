import librosa
import numpy as np

from audio.chord_templates import get_chord_templates


def detect_chord_segments(
    audio_data: dict,
    sensitivity: float = 0.7,
    onset_delta: float = 0.07,
    hop_length: int = 512,
) -> list[dict]:
    """
    Detects chord segments in the audio based on chroma features and onset detection.
    Each segment is returned as a dictionary with keys 'chord', 'start', and 'end' (times in seconds).

    :param audio_data: Dictionary with keys 'data' and 'sr'.
    :param sensitivity: Similarity threshold for chord detection.
    :param onset_delta: Delta parameter for onset detection.
    :param hop_length: Hop length for chroma extraction.
    :return: List of chord segment dictionaries. Segments with low similarity (i.e. "N.C.") are omitted.
    """
    y = audio_data["data"]
    sr = audio_data["sr"]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, delta=onset_delta
    )
    boundaries = [0] + onset_frames.tolist() + [chroma.shape[1]]
    templates = get_chord_templates()
    segments = []
    for i in range(len(boundaries) - 1):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]
        segment_chroma = chroma[:, start_frame:end_frame]
        if segment_chroma.size == 0:
            continue
        avg_chroma = np.mean(segment_chroma, axis=1)
        norm = np.linalg.norm(avg_chroma) + 1e-6
        avg_chroma_norm = avg_chroma / norm
        best_similarity = 0
        best_chord = None
        for chord, template in templates.items():
            template_norm = template / (np.linalg.norm(template) + 1e-6)
            similarity = np.dot(avg_chroma_norm, template_norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_chord = chord
        if best_similarity < sensitivity:
            continue
        start_time_sec = start_frame * hop_length / sr
        end_time_sec = end_frame * hop_length / sr
        segments.append(
            {"chord": best_chord, "start": start_time_sec, "end": end_time_sec}
        )
    return segments
