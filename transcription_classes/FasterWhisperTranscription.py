from faster_whisper import WhisperModel

class FasterWhisperTranscription:
    def __init__(self, audio_path, model_size):
        self.audio_path = audio_path
        self.model_size = model_size

    @staticmethod
    def format_faster_whisper_output(faster_whisper_output):
        # Concatenate all transcriptions into one string
        concatenated_transcription = ' '.join(segment.text for segment in faster_whisper_output)

        return concatenated_transcription


    def transcribe(self):
        print(f"{'Starting transcription for FasterWhisper':-^60}")

        # Run on GPU with INT8 for old GPU
        model = WhisperModel(self.model_size, device="cuda", compute_type="int8")
        
        # or run on GPU with FLOAT16
        # model = WhisperModel(model_size, device="cuda", compute_type="float16")

        segments, _ = model.transcribe(self.audio_path, beam_size=5)

        return self.format_faster_whisper_output(segments)