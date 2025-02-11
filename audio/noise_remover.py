import noisereduce as nr
import numpy as np


def remove_noise(audio_data: dict, level: float = 0.8) -> dict:
    """
    Reduces noise from the audio signal using the noisereduce library.
    If the processed audio contains NaN or Inf values, returns the original audio.

    :param audio_data: Dictionary with 'data' and 'sr'.
    :param level: Noise reduction factor.
    :return: Dictionary with denoised audio data and original sample rate,
             or the original audio data if invalid values are detected.
    """
    y = audio_data["data"]
    sr = audio_data["sr"]
    reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=level)
    if not np.all(np.isfinite(reduced)):
        return audio_data
    return {"data": reduced, "sr": sr}
