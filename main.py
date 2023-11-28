from transcription_classes.FasterWhisperTranscription import FasterWhisperTranscription
from transcription_classes.GladiaTranscription import GladiaTranscription
from transcription_classes.WhisperTranscription import WhisperTranscription

from utils import normalize_transcription, get_audio_duration, transcribe_and_time
from html import escape
from pathlib import Path

import difflib
import argparse
import time

# Function to compare transcriptions
def compare_transcriptions_to_html(transcription1, transcription2):
    # Normalize the transcriptions
    norm_transcription1 = normalize_transcription(transcription1).split()
    norm_transcription2 = normalize_transcription(transcription2).split()

    # Find the differences using difflib
    d = difflib.Differ()
    diff = list(d.compare(norm_transcription1, norm_transcription2))

    # Generate HTML
    html_diff = []
    for word in diff:
        if word.startswith("+ "):
            html_diff.append(f"<span style='color:green;'>{escape(word[2:])}</span>")
        elif word.startswith("- "):
            html_diff.append(f"<span style='color:red;'>{escape(word[2:])}</span>")
        elif word.startswith("  "):
            html_diff.append(escape(word[2:]))

    return ' '.join(html_diff)

def main(args):
    start_time = time.time()

    audio_path = args.audio
    audio_name = Path(audio_path).stem
    model_size = args.model_size
    hf_model = f'openai/whisper-{model_size}'

    audio_duration = get_audio_duration(audio_path)
    print(f"Audio Duration: {audio_duration} seconds")

    # Initialize transcription objects
    faster_whisper = FasterWhisperTranscription(audio_path, model_size)
    gladia = GladiaTranscription(audio_path)
    whisper = WhisperTranscription(audio_path, hf_model, model_size)

    try:
        # Transcribe audio using the services
        transcription_fw, _ = transcribe_and_time(faster_whisper, audio_duration)
        transcription_gladia, _ = transcribe_and_time(gladia, audio_duration)
        transcription_whisper, _ = transcribe_and_time(whisper, audio_duration)
                
        # Compare the transcriptions and write to HTML files
        comparisons = [
            (transcription_fw, transcription_gladia, "fw_and_gladia"),
            (transcription_fw, transcription_whisper, "fw_and_whisper"),
            (transcription_whisper, transcription_gladia, "whisper_and_gladia")
        ]

        Path(f"./outputs/{audio_name}").mkdir(parents=True, exist_ok=True)
        for t1, t2, filename in comparisons:
            result = compare_transcriptions_to_html(t1, t2)
            with open(f"./outputs/{audio_name}/transcription_comparison_{filename}.html", "w") as file:
                file.write(result)

    except Exception as e:
        print(f"Error: {e}") 

    total_end_time = time.time()
    print(f"Total Running Time: {total_end_time - start_time} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--audio", help="name of the target audio file", required=True
    )

    parser.add_argument(
        "--model-size",
        dest="model_size",
        default="large-v3",
        help="name of the Whisper model to use",
    )

    args = parser.parse_args()

    main(args)