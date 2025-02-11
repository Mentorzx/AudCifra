import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import yaml

from audio.chord_detector import detect_chord_segments
from audio.listener import get_audio_data, segments_to_phrases, transcribe_audio_full
from audio.noise_remover import remove_noise
from doc_generator.word_doc_generator import (
    align_chords_to_phrase,
    generate_word_document_chord_lyrics,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    :param config_path: Path to the configuration file.
    :return: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_music(file_path: str, config: dict) -> None:
    """
    Processes an audio file by:
      1. Transcribing the entire audio and grouping segments into phrases based on uppercase.
      2. Detecting chord segments with timestamps.
      3. Associating chords to each phrase and positioning chord labels proportionally.
      4. Adding unassigned chord segments to the last phrase.
      5. Generating a Word document with the desired format.

    :param file_path: Path to the audio file.
    :param config: Configuration dictionary.
    """
    logger.info(f"Processing: {file_path}")
    audio_data = get_audio_data(file_path)
    if config.get("noise_removal", {}).get("enabled", False):
        audio_data = remove_noise(
            audio_data, level=config["noise_removal"].get("level", 0.8)
        )
    transcription_segments = transcribe_audio_full(
        audio_data, model_name=config["transcription"].get("model", "small")
    )
    phrases = segments_to_phrases(transcription_segments)
    chord_segments = detect_chord_segments(
        audio_data,
        sensitivity=config["chord_detection"].get("sensitivity", 0.7),
        onset_delta=config["chord_detection"].get("onset_delta", 0.07),
        hop_length=config["chord_detection"].get("hop_length", 512),
    )
    phrase_chord_data = []
    for phrase in phrases:
        segs_in_phrase = [
            seg
            for seg in chord_segments
            if seg["start"] >= phrase["start"] and seg["start"] <= phrase["end"]
        ]
        chord_line = align_chords_to_phrase(phrase, segs_in_phrase)
        phrase_chord_data.append(
            {
                "lyric": phrase["text"],
                "chord_line": chord_line,
                "start": phrase["start"],
                "end": phrase["end"],
            }
        )
    assigned = []
    for phrase in phrases:
        for seg in chord_segments:
            if seg["start"] >= phrase["start"] and seg["start"] <= phrase["end"]:
                assigned.append(seg)
    unassigned = [seg for seg in chord_segments if seg not in assigned]
    if phrase_chord_data and unassigned:
        last_phrase = phrase_chord_data[-1]
        extra_line = align_chords_to_phrase(
            {
                "start": last_phrase["start"],
                "end": last_phrase["end"],
                "text": " " * len(last_phrase["lyric"]),
            },
            unassigned,
        )
        last_phrase["chord_line"] = (
            last_phrase["chord_line"].rstrip() + " " + extra_line.strip()
        )
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + ".docx"
    output_path = os.path.join(
        config.get("output_folder", "data/outputs/"), output_filename
    )
    generate_word_document_chord_lyrics(phrase_chord_data, output_path)
    logger.info(f"Document generated: {output_path}")


async def main() -> None:
    """
    Main asynchronous function that loads configuration, collects audio files, and processes each file concurrently.
    """
    config = load_config("config.yml")
    audio_folder = config.get("audio_folder", "data/musics/")
    files = [
        os.path.join(audio_folder, f)
        for f in os.listdir(audio_folder)
        if f.lower().endswith((".mp3", ".wav"))
    ]
    logger.info(f"Files found: {files}")
    loop = asyncio.get_running_loop()
    num_cores = os.cpu_count() or 1
    max_workers = max(1, num_cores - 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, process_music, file, config)
            for file in files
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
