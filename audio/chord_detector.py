from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from .chord_classifier import ChordClassifier
from .chord_templates import get_chord_templates
from .segmentation import segment_audio

EPSILON = 1e-6


def mean_representation(segment_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the mean of the frames along the columns.

    Parameters:
        segment_matrix (np.ndarray): A 12 x N matrix of chroma features.

    Returns:
        np.ndarray: A 12-dimensional vector representing the chord.
    """
    return np.mean(segment_matrix, axis=1)


def median_representation(segment_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the median of the frames along the columns.

    Parameters:
        segment_matrix (np.ndarray): A 12 x N matrix of chroma features.

    Returns:
        np.ndarray: A 12-dimensional vector representing the chord.
    """
    return np.median(segment_matrix, axis=1)


def weighted_representation(segment_matrix: np.ndarray) -> np.ndarray:
    """
    Returns a weighted representation of the frames using the energy (norm) of each frame.

    Parameters:
        segment_matrix (np.ndarray): A 12 x N matrix of chroma features.

    Returns:
        np.ndarray: A 12-dimensional vector representing the chord.
    """
    energies = np.linalg.norm(segment_matrix, axis=0)
    total = np.sum(energies) + EPSILON
    return np.sum(segment_matrix * energies, axis=1) / total


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Parameters:
        x (np.ndarray): First normalized vector.
        y (np.ndarray): Second normalized vector.

    Returns:
        float: The cosine similarity score.
    """
    return float(np.dot(x, y))


def euclidean_similarity(x: np.ndarray, y: np.ndarray):
    """
    Computes a similarity score based on the Euclidean distance.

    Parameters:
        x (np.ndarray): First normalized vector.
        y (np.ndarray): Second normalized vector.

    Returns:
        float: The similarity score, inversely proportional to the distance.
    """
    dist = np.linalg.norm(x - y)
    return 1.0 / (1.0 + dist)


def correlation_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient as a similarity measure.

    Parameters:
        x (np.ndarray): First normalized vector.
        y (np.ndarray): Second normalized vector.

    Returns:
        float: The correlation coefficient, or 0 if variance is too low.
    """
    if np.std(x) < EPSILON or np.std(y) < EPSILON:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


class ChordDetector:
    """
    Detects chords in audio using adaptive segmentation and either template matching or a trained classifier.
    It provides robust chroma representations with configurable aggregation and similarity metrics, and applies
    temporal smoothing to enforce musical continuity.
    """

    def __init__(
        self,
        sensitivity: float = 0.7,
        onset_delta: float = 0.07,
        hop_length: int = 512,
        use_classifier: bool = False,
        representation_method: str = "median",  # options: "mean", "median", "weighted"
        similarity_metric: str = "cosine",  # options: "cosine", "euclidean", "correlation"
        use_sliding_window: bool = False,
        window_size: int = 10,
        window_overlap: float = 0.5,
        smoothing: bool = True,
    ):
        """
        Initializes the chord detector with configuration parameters.

        Parameters:
            sensitivity (float): Similarity threshold to accept a chord.
            onset_delta (float): Delta value for onset detection.
            hop_length (int): Hop length for chroma extraction.
            use_classifier (bool): Whether to use a trained classifier.
            representation_method (str): Method to aggregate frames ("mean", "median", or "weighted").
            similarity_metric (str): Similarity metric ("cosine", "euclidean", or "correlation").
            use_sliding_window (bool): Whether to use overlapping sliding windows for robustness.
            window_size (int): Number of frames per sliding window.
            window_overlap (float): Fractional overlap between consecutive windows.
            smoothing (bool): Apply temporal smoothing to the chord sequence.
        """
        self.sensitivity = sensitivity
        self.onset_delta = onset_delta
        self.hop_length = hop_length
        self.use_classifier = use_classifier
        self.representation_method = representation_method.lower()
        self.similarity_metric = similarity_metric.lower()
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.smoothing = smoothing
        self.epsilon = EPSILON

        self.templates = get_chord_templates(False, True)
        self.classifier = ChordClassifier(None) if self.use_classifier else None

        self.representation_funcs: Dict[str, Any] = {
            "mean": mean_representation,
            "median": median_representation,
            "weighted": weighted_representation,
        }
        self.similarity_funcs: Dict[str, Any] = {
            "cosine": cosine_similarity,
            "euclidean": euclidean_similarity,
            "correlation": correlation_similarity,
        }

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalizes the vector to have unit norm.

        Parameters:
            vector (np.ndarray): Input vector.

        Returns:
            np.ndarray: Normalized vector.
        """
        norm = np.linalg.norm(vector) + self.epsilon
        return vector / norm

    def compute_representation(self, segment_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the chroma representation for a segment using the selected method.

        Parameters:
            segment_matrix (np.ndarray): A 12 x N matrix of chroma features.

        Returns:
            np.ndarray: A 12-dimensional vector representing the chord.
        """
        func = self.representation_funcs.get(
            self.representation_method, mean_representation
        )
        return func(segment_matrix)

    def compute_sliding_representation(self, segment_matrix: np.ndarray) -> np.ndarray:
        """
        Computes a robust representation using overlapping sliding windows.

        Parameters:
            segment_matrix (np.ndarray): A 12 x N matrix of chroma features.

        Returns:
            np.ndarray: A 12-dimensional vector representing the chord.
        """
        n_frames = segment_matrix.shape[1]
        step = max(int(self.window_size * (1 - self.window_overlap)), 1)
        reps = [
            self.compute_representation(segment_matrix[:, i : i + self.window_size])
            for i in range(0, n_frames - self.window_size + 1, step)
        ]
        if reps:
            return np.median(np.stack(reps, axis=0), axis=0)
        return self.compute_representation(segment_matrix)

    def _classify_chord(self, rep_norm: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classifies the chord using either a trained classifier or template matching.

        Parameters:
            rep_norm (np.ndarray): Normalized chroma representation.

        Returns:
            Tuple[Optional[str], float]: The chord label and its similarity score.
        """
        if self.use_classifier and self.classifier:
            return self.classifier.predict(rep_norm)
        best_similarity = -np.inf
        best_chord: Optional[str] = None
        sim_func = self.similarity_funcs.get(self.similarity_metric, cosine_similarity)
        for chord, template in self.templates.items():
            template_norm = self._normalize(template)
            similarity = sim_func(rep_norm, template_norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_chord = chord

        return best_chord, best_similarity

    def smooth_chord_sequence(
        self, segments: List[Dict[str, Any]], window_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Applies a mode-based smoothing filter to the chord sequence for temporal continuity.

        Parameters:
            segments (List[Dict[str, Any]]): List of segment dictionaries with a "chord" key.
            window_size (int): Number of segments to consider for smoothing (should be odd).

        Returns:
            List[Dict[str, Any]]: Smoothed list of chord segments.
        """
        labels = [seg["chord"] for seg in segments]
        smoothed_labels = labels.copy()
        for i in range(1, len(labels) - 1):
            window = labels[i - 1 : i + 2]
            smoothed_labels[i] = max(set(window), key=window.count)
        for i, seg in enumerate(segments):
            seg["chord"] = smoothed_labels[i]
        return segments

    def detect(self, audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects chord segments in the given audio data using adaptive segmentation and robust chord recognition.

        Parameters:
            audio_data (Dict[str, Any]): Dictionary containing 'data' (audio signal) and 'sr' (sample rate).

        Returns:
            List[Dict[str, Any]]: List of segments with chord labels and start/end times (in seconds).
        """
        segments = segment_audio(audio_data, self.hop_length, self.onset_delta, None)
        y = audio_data["data"]
        sr = audio_data["sr"]
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        detected_segments: List[Dict[str, Any]] = []
        for segment in segments:
            start_frame = segment["start_frame"]
            end_frame = segment["end_frame"]
            segment_chroma = chroma[:, start_frame:end_frame]
            if segment_chroma.size == 0:
                continue
            if self.use_sliding_window and segment_chroma.shape[1] >= self.window_size:
                rep = self.compute_sliding_representation(segment_chroma)
            else:
                rep = self.compute_representation(segment_chroma)
            rep_norm = self._normalize(rep)
            chord_label, similarity = self._classify_chord(rep_norm)
            if not chord_label or similarity < self.sensitivity:
                continue
            if detected_segments and detected_segments[-1]["chord"] == chord_label:
                continue
            start_time = start_frame * self.hop_length / sr
            end_time = end_frame * self.hop_length / sr
            detected_segments.append(
                {"chord": chord_label, "start": start_time, "end": end_time}
            )
        # if self.smoothing and len(detected_segments) >= 3:
        #     detected_segments = self.smooth_chord_sequence(detected_segments)

        return detected_segments


def detect_chord_segments(
    audio_data: Dict[str, Any],
    sensitivity: float = 0.7,
    onset_delta: float = 0.07,
    hop_length: int = 512,
) -> List[Dict[str, Any]]:
    """
    Convenience function that instantiates a ChordDetector and returns the detected chord segments.

    Parameters:
        audio_data (Dict[str, Any]): Dictionary containing 'data' and 'sr'.
        sensitivity (float): Similarity threshold for detection.
        onset_delta (float): Delta for onset detection.
        hop_length (int): Hop length for chroma extraction.

    Returns:
        List[Dict[str, Any]]: List of chord segment dictionaries with keys 'chord', 'start', and 'end'.
    """
    detector = ChordDetector(sensitivity, onset_delta, hop_length)
    return detector.detect(audio_data)
