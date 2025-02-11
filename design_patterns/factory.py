from typing import Callable

from audio.chord_detector import detect_chord_segments
from audio.noise_remover import remove_noise


def get_audio_processor(processor_type: str) -> Callable:
    """
    Returns an audio processing function based on the processor type.

    :param processor_type: The type of processor ("noise" or "chord").
    :return: The corresponding processing function.
    """
    processors: dict[str, Callable] = {
        "noise": remove_noise,
        "chord": detect_chord_segments,
    }
    return processors.get(processor_type, lambda: None)
