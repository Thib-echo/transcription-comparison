from transcription_classes.FasterWhisperTranscription import FasterWhisperTranscription
from transcription_classes.GladiaTranscription import GladiaTranscription
from transcription_classes.WhisperTranscription import WhisperTranscription
from transcription_classes.StableWhisperTranscription import StableWhisperTranscription

from utils import normalize_transcription, get_audio_duration, transcribe_and_time
from html import escape
from pathlib import Path

import difflib
import argparse
import time

# Function to compare transcriptions
def compare_transcriptions_to_html(transcription1, transcription2, color1='blue', color2='green'):
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
            html_diff.append(f"<span style='color:{color2};'>{escape(word[2:])}</span>")
        elif word.startswith("- "):
            html_diff.append(f"<span style='color:{color1};'>{escape(word[2:])}</span>")
        elif word.startswith("  "):
            html_diff.append(escape(word[2:]))

    return ' '.join(html_diff)

def generate_comparison_matrix_html(transcriptions, tool_names):
    html_output = "<table border='1'><tr><th></th>"
    color = {"FasterWhisper": 'red',
             "Gladia": 'purple', 
             "Whisper": 'green', 
             "StableWhisper": 'blue'
    }

    # Table headers
    for name in tool_names:
        html_output += f"<th>{name}</th>"
    html_output += "</tr>"

    for i, (name1, transcription1) in enumerate(zip(tool_names, transcriptions)):
        html_output += f"<tr><th>{name1}</th>"
        for j, (name2, transcription2) in enumerate(zip(tool_names, transcriptions)):
            if i != j:
                # Compare and create a cell with highlighted differences
                comparison_html = compare_transcriptions_to_html(transcription1, transcription2, color1=color[name1], color2=color[name2])
                html_output += f"<td>{comparison_html}</td>"
            else:
                html_output += "<td style='background-color:lightgrey;'></td>"  # Empty cell for same-tool comparison
        html_output += "</tr>"

    html_output += "</table>"
    return html_output

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
    stable_whisper = StableWhisperTranscription(audio_path, hf_model, model_size)

    try:
        # Transcribe audio using the services
        transcription_fw, _ = transcribe_and_time(faster_whisper, audio_duration)
        transcription_gladia, _ = transcribe_and_time(gladia, audio_duration)
        transcription_whisper, _ = transcribe_and_time(whisper, audio_duration)
        transcription_sw, _ = transcribe_and_time(stable_whisper, audio_duration)

        transcriptions = [transcription_fw, transcription_gladia, transcription_whisper, transcription_sw]
        tool_names = ["FasterWhisper", "Gladia", "Whisper", "StableWhisper"]

        # Generate comparison matrix
        comparison_matrix_html = generate_comparison_matrix_html(transcriptions, tool_names)

        # Save the comparison matrix to an HTML file
        output_filename = f"./outputs/{audio_name}/transcription_comparison_matrix.html"
        Path(output_filename).parents[0].mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w") as file:
            file.write(comparison_matrix_html)

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