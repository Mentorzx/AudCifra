from typing import Optional

import librosa
import numpy as np


def segment_audio(
    audio_data: dict,
    hop_length: int,
    onset_delta: float,
    diff_threshold: Optional[float],
) -> list:
    """
    Segments the audio into regions based on both onset detection and harmonic change detection.

    This function computes chroma features and calculates the difference between consecutive frames.
    It then combines the detected onsets with frames where the chroma difference exceeds an adaptive
    threshold to identify boundaries where harmonic changes occur. This adaptive segmentation aims to
    capture chord changes more accurately.

    Parameters:
        audio_data (dict): A dictionary containing keys 'data' and 'sr'.
        hop_length (int): The hop length for feature extraction.
        onset_delta (float): The delta parameter for onset detection.
        diff_threshold (float, optional): A threshold for chroma differences. If not provided, it is computed as
            1.5 times the median difference between consecutive chroma frames.

    Returns:
        list: A list of dictionaries, each with 'start_frame' and 'end_frame' keys indicating segment boundaries.
    """
    y = audio_data["data"]
    sr = audio_data["sr"]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    n_frames = chroma.shape[1]
    chroma_diff = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
    if not diff_threshold:
        median_diff = np.median(chroma_diff)
        diff_threshold = 1.5 * median_diff
    diff_boundaries = np.where(chroma_diff > diff_threshold)[0] + 1
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, delta=onset_delta
    )
    combined_boundaries = np.union1d(diff_boundaries, onset_frames)
    all_boundaries = np.concatenate(([0], combined_boundaries, [n_frames]))
    all_boundaries = np.unique(all_boundaries)
    all_boundaries = np.sort(all_boundaries)
    segments = []
    segments.extend(
        {
            "start_frame": int(all_boundaries[i]),
            "end_frame": int(all_boundaries[i + 1]),
        }
        for i in range(len(all_boundaries) - 1)
    )
    return segments
