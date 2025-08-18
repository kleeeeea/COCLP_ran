import os
import torch
import json
import io
import numpy as np
import sys
from opencc import OpenCC
from pydub import AudioSegment
import hashlib
from faster_whisper import WhisperModel

cc = OpenCC('t2s')  # 繁体转简体

class WhisperNoVADTranscribeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_dir": ("STRING", {"default": "./audio/"}),
                "model_path": ("STRING", {"default": "./faster-whisper"}),
                "output_dir": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": "all_transcripts.jsonl"}),
                "language": ("STRING", {"default": "zh"}),
                "beam_size": ("INT", {"default": 5}),
                "device": ("STRING", {"default": "cuda"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "MY_NODES/Whisper"

    def _hash_path_to_id(self, file_path: str) -> str:
        """Stable SHA‑256 hash so each Markdown file gets a unique ID."""
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()

    def load_audio_bytes(self, wav_bytes, sr=16000):
        try:
            audio = AudioSegment.from_file(io.BytesIO(wav_bytes))
            audio = audio.set_channels(1).set_frame_rate(sr)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2 ** 15)
            return samples, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

    def run(self, input_dir, model_path, output_dir, language, beam_size, device, output_filename):
        model = WhisperModel(model_path, device=device, compute_type="int8")
        valid_exts = [".wav", ".mp3", ".m4a", ".flac", ".aac"]
        os.makedirs(output_dir, exist_ok=True)

        audio_files = [f for f in sorted(os.listdir(input_dir)) if any(f.endswith(ext) for ext in valid_exts)]
        if not audio_files:
            return ("No valid audio files found.",)

        output_jsonl_path = os.path.join(output_dir, output_filename)
        with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
            for fname in audio_files:
                file_path = os.path.join(input_dir, fname)
                file_id = os.path.splitext(fname)[0]
                try:
                    with open(file_path, "rb") as f:
                        wav_bytes = f.read()

                    y, sr = self.load_audio_bytes(wav_bytes, sr=16000)
                    segments, _ = model.transcribe(
                        y,
                        language=language,
                        beam_size=beam_size,
                        word_timestamps=False,
                        condition_on_previous_text=True
                    )

                    id=self._hash_path_to_id(file_path)

                    full_text = cc.convert("".join([seg.text for seg in segments]).strip())
                    json.dump({"id":id, "audio_path":file_path, "audio_id": file_id, "text": full_text}, out_f, ensure_ascii=False)
                    out_f.write("\n")

                except Exception as e:
                    json.dump({"id":id, "audio_path":file_path, "audio_id": file_id, "error": str(e)}, out_f, ensure_ascii=False)
                    out_f.write("\n")

        return (output_dir,)
