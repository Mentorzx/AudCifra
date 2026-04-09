# AudCifra

AudCifra is an audio-to-document pipeline that transcribes songs, detects chord segments and exports a Word document with aligned lyrics and harmony.

The project combines audio preprocessing, transcription, chord detection and document generation in a single workflow. It is a good example of practical applied AI code where the interesting part is not just the model, but the whole pipeline from raw input to usable output.

## What it does

- loads `.mp3` and `.wav` files
- optionally removes noise before analysis
- transcribes vocals into text segments
- detects chord segments from the audio
- aligns chords to lyric phrases
- generates `.docx` output files
- processes multiple files concurrently with `asyncio` and `ThreadPoolExecutor`

## Stack

- Python 3.12
- librosa
- PyYAML
- python-docx
- asyncio
- custom modules for audio processing and document generation

## Project structure

```text
audio/               # transcription, listening and chord detection
doc_generator/       # Word export helpers
utils/               # logging and shared utilities
config.yml           # runtime configuration
main.py              # pipeline entrypoint
```

## Installation

### Clone the repository

```bash
git clone https://github.com/Mentorzx/AudCifra
cd AudCifra
```

### Install dependencies with uv

```bash
uv sync
```

## Configuration

The pipeline is driven by `config.yml`.

Example:

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

## Run

```bash
uv run python main.py
```

Input files are read from the configured `audio_folder`, and generated `.docx` files are written to the configured `output_folder`.

## Why it matters

AudCifra shows an end-to-end pipeline mindset:

- raw signal processing
- transcription and alignment logic
- concurrent execution
- structured document output instead of just a demo prediction

## Contact

Public profile: [github.com/Mentorzx](https://github.com/Mentorzx)
