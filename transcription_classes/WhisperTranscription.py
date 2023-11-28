from dotenv import load_dotenv
from os import environ as env
import logging
from transformers import (
    pipeline,
    WhisperTokenizer,
    WhisperFeatureExtractor
    )
import soundfile as sf
import librosa
from utils import get_torch_device
import whisper

load_dotenv()

class WhisperTranscription:
    def __init__(self, audio_path, hf_model, model_size):
        self.audio_path = audio_path
        self.hf_model = hf_model
        self.model_size = model_size
    # def transcribe(self):
    #     print("Starting transcription for Whisper")
    #     try:
    #         feature_extractor = WhisperFeatureExtractor.from_pretrained(self.hf_model)
    #         tokenizer = WhisperTokenizer.from_pretrained(
    #             self.hf_model, language="french", task="transcribe"
    #         )

    #         forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
    #             language="french", task="transcribe"
    #         )

    #         pipe = pipeline(
    #             "automatic-speech-recognition",
    #             model=self.hf_model,
    #             feature_extractor=feature_extractor,
    #             tokenizer=tokenizer,
    #             chunk_length_s=30,
    #             device=get_torch_device(),
    #             stride_length_s=(4, 2),
    #         )

    #         # Read and resample audio file
    #         audio_data, original_sample_rate = sf.read(self.audio_path)
    #         if original_sample_rate != 16000:
    #             audio_data = librosa.resample(
    #                 audio_data, orig_sr=original_sample_rate, target_sr=16000
    #             )

    #         prediction = pipe(
    #             audio_data,
    #             generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
    #             batch_size=8,
    #         )["text"]

    #         return prediction

    #     except Exception as e:
    #         logging.error(f"Transcription failed: {e}")
    #         return None

    def transcribe(self):
        print(f"{'Starting transcription for Whisper':-^60}")
        model = whisper.load_model(self.model_size)
        result = model.transcribe(self.audio_path, language='fr', beam_size=5)
        
        return result["text"]