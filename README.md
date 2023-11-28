
# Audio Transcription Comparison Tool

## Overview
This tool is designed to transcribe audio files using different transcription services and compare their outputs. It currently supports FasterWhisper, Gladia, and Whisper transcription services. The tool calculates the transcription time for each service, compares the transcribed texts, and provides a visual representation of the differences in HTML format.

## Features
- Transcribe audio files using FasterWhisper, Gladia, and Whisper.
- Time each transcription process.
- Compare transcriptions and highlight differences.
- Generate HTML files to visualize the comparison.

## Installation

### Prerequisites
- Python 3.11 or higher
- Pip package manager

### Setting Up
Clone the repository or download the source code:

```bash
git clone git@github.com:Thib-echo/transcription-comparison.git
cd transcription-comparison
```
Create a virtual env and activate it:
```bash
python -m venv env
source env/bin/activate # For linux users
.\env\Script\activate # For windows users
```

### Dependencies
Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Tokens
In order to use GladiaTranscription, you need to create an account on [Gladia](https://www.gladia.io/) and then create an API key.  
That key can be used as it in the GladiaTranscription class.  
You can also use `python-dotenv` :
```bash
pip install python-dotenv
```
And then create an `.env` file that will be filed with the token :
```bash
GLADIA_TOKEN = "YOUR_TOKEN"
```

### Basic Command
Run the script from the command line with the following arguments:

```bash
python main.py -a [path to audio file] --model-size [model size]
```

- `-a, --audio`: Path to the target audio file.
- `--model-size`: Name of the Whisper model to use (e.g., `large-v3`).

### Example
```bash
python main.py -a ./audio_files/1.wav --model-size large-v3
```

This will transcribe the provided audio file using all three services, compare their outputs, and generate HTML files with the comparisons.

## Output
The script prints the transcription time for each service and the total running time to the console. It also creates HTML files in the outputs directory, visualizing the differences between each pair of transcription services.