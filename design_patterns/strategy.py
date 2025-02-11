from typing import Any, Protocol

from audio.noise_remover import remove_noise


class NoiseRemovalStrategy(Protocol):
    """
    Protocol for noise removal strategies.

    :return: Processed audio data.
    """

    def remove_noise(self, audio_data: Any, level: float) -> Any: ...


class BasicNoiseRemoval:
    """
    Basic noise removal strategy using the default remove_noise function.

    :return: Processed audio data.
    """

    def remove_noise(self, audio_data: Any, level: float) -> Any:
        return remove_noise(audio_data, level)


class AdvancedNoiseRemoval:
    """
    Advanced noise removal strategy with a custom implementation.

    :return: Processed audio data.
    """

    def remove_noise(self, audio_data: Any, level: float) -> Any:
        import noisereduce as nr

        y = audio_data["data"]
        sr = audio_data["sr"]
        reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=level)
        return {"data": reduced, "sr": sr}
