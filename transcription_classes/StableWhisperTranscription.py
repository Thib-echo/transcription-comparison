import stable_whisper

class StableWhisperTranscription:
    def __init__(self, audio_path, hf_model, model_size):
        self.audio_path = audio_path
        self.hf_model = hf_model
        self.model_size = model_size

    def transcribe(self):
        print(f"{'Starting transcription for StableWhisper':-^60}")
        model = stable_whisper.load_faster_whisper(self.model_size, device="cuda", compute_type="int8")
        result = model.transcribe_stable(self.audio_path, language='fr', beam_size=5)
        
        return result.to_txt()
