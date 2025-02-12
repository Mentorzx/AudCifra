# Audio Chord Transcriber

**Audio Chord Transcriber** is a project that transforms audio files into Word documents containing chords and lyrics. It performs audio transcription, detects chord segments, aligns chords to corresponding phrases, and generates a final document with both chord and lyric information.

## Features

- **Audio Extraction:**  
  Loads and processes audio files (`.mp3`, `.wav`).

- **Optional Noise Removal:**  
  Enhances audio quality by removing unwanted noise.

- **Audio Transcription:**  
  Converts audio into transcription segments.

- **Chord Detection:**  
  Analyzes the audio to identify chord segments.

- **Chord & Lyric Alignment:**  
  Associates the detected chords with the respective phrases in the song.

- **Word Document Generation:**  
  Creates a `.docx` file with aligned chords and lyrics.

- **Flexible Configuration:**  
  Choose whether to merge instrumental (non-vocal) segments with the previous vocal segment or ignore them, via the `config.yml`.

- **Parallel Processing:**  
  Utilizes `asyncio` and `ThreadPoolExecutor` to process multiple files concurrently.

## Requirements

- Python 3.12 or higher
- Dependencies as listed in `pyproject.toml`
- [Ultra Violet (uv)](https://astral.sh/uv/) for environment and dependency management

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Mentorzx/AudCifra
   cd AudCifra
   ```

2. **Install Ultra Violet (uv):**

    - curl -LsSf https://astral.sh/uv/install.sh | sh
    Note: On Windows, use Git Bash or WSL to execute the command above.

3. **uv install**

## Configuration
    The project uses a config.yml file to manage various settings. Below is an example configuration:

    ```yaml
    audio_folder: "data/musics/"
    output_folder: "data/outputs/"
    merge_instrumental: true
    transcription:
        model: "small"
    logger:
        console_level: "WARNING"
        file_level: "DEBUG"
    noise_removal:
        enabled: true
        level: 0.4
    chord_detection:
        sensitivity: 0.9
        onset_delta: 0.5
        hop_length: 512
    ```