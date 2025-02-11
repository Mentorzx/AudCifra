import numpy as np


def get_chord_templates() -> dict:
    """
    Generates chord templates for various chord types (maj, min, 7, maj7, m7, sus2, sus4, add9, madd9, 9).
    Each template is a binary vector of length 12 corresponding to the 12 pitch classes.

    :return: Dictionary mapping chord names to their binary templates.
    """
    templates = {}
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chord_types = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "7": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "add9": [0, 4, 7, 2],
        "madd9": [0, 3, 7, 2],
        "9": [0, 4, 7, 10, 2],
    }
    display_names = {
        "maj": "",
        "min": "m",
        "7": "7",
        "maj7": "maj7",
        "min7": "m7",
        "sus2": "sus2",
        "sus4": "sus4",
        "add9": "add9",
        "madd9": "madd9",
        "9": "9",
    }
    for i, note in enumerate(notes):
        for chord_key, intervals in chord_types.items():
            template = [0] * 12
            for interval in intervals:
                template[(i + interval) % 12] = 1
            chord_name = note + display_names[chord_key]
            templates[chord_name] = np.array(template, dtype=float)
    return templates
