import os
from cog import BasePredictor, Input, Path
from pyngrok import ngrok, conf

import sys
sys.path.append('/content/metavoice-src')
os.chdir('/content/metavoice-src')

import io
import json
import requests
import soundfile as sf

API_SERVER_URL = "http://127.0.0.1:58003/tts"
RADIO_CHOICES = ["Preset voices", "Upload target voice"]
MAX_CHARS = 220
PRESET_VOICES = {
    # female
    "Ava": "https://cdn.themetavoice.xyz/speakers/ava.flac",
    "Bria": "https://cdn.themetavoice.xyz/speakers/bria.mp3",
    # male
    "Alex": "https://cdn.themetavoice.xyz/speakers/alex.mp3",
    "Jacob": "https://cdn.themetavoice.xyz/speakers/jacob.wav",
}

def denormalise_top_p(top_p):
    # returns top_p in the range [0.9, 1.0]
    return round(0.9 + top_p / 100, 2)

def denormalise_guidance(guidance):
    # returns guidance in the range [1.0, 3.0]
    return 1 + ((guidance - 1) * (3 - 1)) / (5 - 1)

def _handle_edge_cases(to_say, upload_target):
    if not to_say:
        raise gr.Error("Please provide text to synthesise")
    if len(to_say) > MAX_CHARS:
        gr.Warning(
            f"Max {MAX_CHARS} characters allowed. Provided: {len(to_say)} characters. Truncating and generating speech...Result at the end can be unstable as a result."
        )
    def _check_file_size(path):
        if not path:
            return
        filesize = os.path.getsize(path)
        filesize_mb = filesize / 1024 / 1024
        if filesize_mb >= 50:
            raise gr.Error(
                f"Please upload a sample less than 20MB for voice cloning. Provided: {round(filesize_mb)} MB"
            )
    _check_file_size(upload_target)

def tts(to_say, top_p, guidance, toggle, preset_dropdown, upload_target):
    d_top_p = denormalise_top_p(top_p)
    d_guidance = denormalise_guidance(guidance)
    _handle_edge_cases(to_say, upload_target)
    to_say = to_say if len(to_say) < MAX_CHARS else to_say[:MAX_CHARS]
    custom_target_path = upload_target if toggle == RADIO_CHOICES[1] else None
    config = {
        "text": to_say,
        "guidance": d_guidance,
        "top_p": d_top_p,
        "speaker_ref_path": PRESET_VOICES[preset_dropdown] if toggle == RADIO_CHOICES[0] else None,
    }
    headers = {"Content-Type": "audio/wav", "X-Payload": json.dumps(config)}
    if not custom_target_path:
        response = requests.post(API_SERVER_URL, headers=headers, data=None)
    else:
        with open(custom_target_path, "rb") as f:
            data = f.read()
            response = requests.post(API_SERVER_URL, headers=headers, data=data)
    wav, sr = None, None
    if response.status_code == 200:
        audio_buffer = io.BytesIO(response.content)
        audio_buffer.seek(0)
        wav, sr = sf.read(audio_buffer, dtype="float32")
    else:
        print(f"Something went wrong. response status code: {response.status_code}")
    return sr, wav

def change_voice_selection_layout(choice):
    if choice == RADIO_CHOICES[0]:
        return [gr.update(visible=True), gr.update(visible=False)]
    return [gr.update(visible=False), gr.update(visible=True)]

class Predictor(BasePredictor):
    def setup(self) -> None:
        os.system(f"pip install -e .")
        import threading
        import subprocess
        def run_command_in_thread(command):
            subprocess.run(command, shell=True)
        command = 'cd /content/metavoice-src && python fam/llm/serving.py --huggingface_repo_id="metavoiceio/metavoice-1B-v0.1"'
        thread = threading.Thread(target=run_command_in_thread, args=(command,))
        thread.start()
    def predict(
        self,
        input_audio: Path = Input(description="Input Image"),
        to_say: str = Input(default="This is a demo of text to speech by MetaVoice-1B, an open-source foundational audio model by MetaVoice."),
    ) -> Path:
        audio_sr, audio_wav = tts(to_say, 5.0, 5.0, RADIO_CHOICES[0], list(PRESET_VOICES.keys())[0], input_audio)
        import soundfile as sf
        sf.write('/content/audio.wav', audio_wav, audio_sr)
        return Path('/content/audio.wav')