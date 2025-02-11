import noisereduce as nr


def remove_noise(audio_data: dict, level: float = 0.8) -> dict:
    """
    Removes noise from the audio data using noisereduce.

    :param audio_data: Dictionary with audio data.
    :param level: Proportional decrease factor for noise reduction.
    :return: Processed audio data.
    """
    y = audio_data["data"]
    sr = audio_data["sr"]
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=level)
    return {"data": reduced_noise, "sr": sr}
