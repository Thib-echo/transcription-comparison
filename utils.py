import torch
import re
import librosa
import time

def transcribe_and_time(transcription_service, audio_duration):
    start_time = time.time()
    transcription = transcription_service.transcribe()
    end_time = time.time()

    transcription_time = end_time - start_time
    percentage = (transcription_time / audio_duration) * 100

    print(f"{transcription_service.__class__.__name__} Transcription Time: {transcription_time:.2f} seconds ({percentage:.2f}% of audio duration)")
    
    return transcription, transcription_time


def get_audio_duration(audio_path):
    duration = librosa.get_duration(filename=audio_path)
    return duration

def normalize_transcription(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_torch_device():
    # Check for CUDA
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device("cuda")

    # Check for MPS (requires newer versions of PyTorch)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")

    # Default to CPU
    else:
        print("Using CPU")
        return torch.device("cpu")