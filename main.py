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


def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file and returns the configuration as a dictionary.

    Parameters:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_music(file_path: str, config: dict, logger) -> None:
    """
    Processes a music file by extracting audio data, performing optional noise removal,
    transcribing the audio, detecting chord segments, aligning chords to corresponding phrases,
    and generating a Word document with the chord and lyric data.

    Parameters:
        file_path (str): The path to the music file.
        config (dict): The configuration dictionary.
        logger: A logger instance for logging messages.
    """
    logger.debug(f"Processing: {file_path}")
    audio_data = get_audio_data(file_path)
    if config.get("noise_removal", {}).get("enabled", False):
        audio_data = remove_noise(
            audio_data, level=config["noise_removal"].get("level", 0.4)
        )
    transcription_segments = transcribe_audio_full(
        audio_data, model_name=config["transcription"].get("model", "base")
    )
    phrases = segments_to_phrases(transcription_segments)
    chord_segments = detect_chord_segments(
        audio_data,
        sensitivity=config["chord_detection"].get("sensitivity", 0.9),
        onset_delta=config["chord_detection"].get("onset_delta", 0.5),
        hop_length=config["chord_detection"].get("hop_length", 512),
    )
    if phrases:
        first_phrase_start = phrases[0]["start"]
        chord_segments = [
            seg for seg in chord_segments if seg["start"] >= first_phrase_start
        ]
    phrase_chord_data = []
    for phrase in phrases:
        segs_in_phrase = [
            seg
            for seg in chord_segments
            if phrase["start"] <= seg["start"] <= phrase["end"]
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
    merge_instrumental = config.get("merge_instrumental", True)
    assigned = []
    for phrase in phrases:
        assigned.extend(
            seg
            for seg in chord_segments
            if phrase["start"] <= seg["start"] <= phrase["end"]
        )
    unassigned = [seg for seg in chord_segments if seg not in assigned]
    if merge_instrumental and phrase_chord_data and unassigned:
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
    output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.docx"
    output_path = os.path.join(
        config.get("output_folder", "data/outputs/"), output_filename
    )
    generate_word_document_chord_lyrics(phrase_chord_data, output_path)
    logger.info(f"Document generated: {output_path}")


async def main() -> None:
    """
    Main asynchronous function that loads the configuration, sets up logging,
    locates all audio files in the specified folder, and processes each music file in parallel
    using a ThreadPoolExecutor.

    This function uses asyncio to run the processing tasks concurrently.
    """
    config = load_config("config.yml")
    logger_config = config.get("logger", {})
    console_level = logger_config.get("console_level", "WARNING")
    file_level = logger_config.get("file_level", "DEBUG")
    logger = get_logger(__name__, console_level=console_level, file_level=file_level)
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
            loop.run_in_executor(executor, process_music, file, config, logger)
            for file in files
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
