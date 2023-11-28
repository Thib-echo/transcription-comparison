from dotenv import load_dotenv
from os import environ as env
import requests
from pathlib import Path
import logging

load_dotenv()

class GladiaTranscription:
    def __init__(self, audio_path,):
        self.audio_path = audio_path
        self.gladia_token = env[
            "GLADIA_TOKEN"
        ]

    @staticmethod
    def format_gladia_output(gladia_output):
        # Extract the list of transcriptions
        predictions = gladia_output.get('prediction', [])

        # Concatenate all transcriptions into one string
        concatenated_transcription = ' '.join(prediction['transcription'] for prediction in predictions)

        return concatenated_transcription

    def transcribe(self):
        print("Starting transcription for Gladia")
        headers = {
            "x-gladia-key": self.gladia_token,
            "accept": "application/json",
        }

        file_extension = Path(self.audio_path).suffix[1:]
        file_name = Path(self.audio_path).stem

        with open(self.audio_path, "rb") as f:
            files = {
                "audio": (file_name, f, f"audio/{file_extension}"),
                "toggle_diarization": (None, False),
                # "diarization_max_speakers": (None, self.nb_speaker),
            }

            try:
                response = requests.post(
                    "https://api.gladia.io/audio/text/audio-transcription/",
                    headers=headers,
                    files=files,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logging.error(f"Gladia API Request failed: {e}")
                return None

            return self.format_gladia_output(response.json())