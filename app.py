from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download
from kokoro import KPipeline

VOICES = {
    "Heart — female": "af_heart",
    "Bella — female": "af_bella",
    "Michael — male": "am_michael",
    "Puck — male": "am_puck",
}

OUTPUT = Path(__file__).resolve().parent / "recordings"
OUTPUT.mkdir(exist_ok=True)

print("Loading Kokoro. First launch downloads the model and voices.")
pipeline = KPipeline(lang_code="a", device="cpu")

# Download every offered voice now so each works offline later.
for voice in VOICES.values():
    hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename=f"voices/{voice}.pt",
    )

def generate(text, voice_name, speed):
    text = text.strip()
    if not text:
        raise gr.Error("Enter the text you want spoken.")

    chunks = []
    for _, _, audio in pipeline(
        text, voice=VOICES[voice_name], speed=float(speed)
    ):
        if audio is not None:
            chunks.append(audio.cpu().numpy())

    if not chunks:
        raise gr.Error("No audio was generated.")

    audio = np.concatenate(chunks)
    # Kokoro generates 24 kHz. Save mono, 16 kHz, 16-bit PCM
    # for our STM32 audio workflow.
    audio = resample_poly(audio, 2, 3)
    path = OUTPUT / f"speech_{uuid4().hex[:10]}.wav"
    sf.write(path, audio, 16000, subtype="PCM_16")
    return str(path), str(path)

with gr.Blocks(title="Offline Voice Generator") as app:
    gr.Markdown("# Offline Voice Generator")
    text = gr.Textbox(
        label="Text to speak",
        lines=5,
        value="I detect one person in the camera view.",
    )
    voice = gr.Dropdown(
        choices=list(VOICES),
        value="Heart — female",
        label="Voice",
    )
    speed = gr.Slider(
        minimum=0.7, maximum=1.3, value=1.0,
        step=0.05, label="Speed",
    )
    button = gr.Button("Generate speech", variant="primary")
    preview = gr.Audio(label="Listen", type="filepath")
    download = gr.File(label="Download WAV")
    button.click(
        generate,
        inputs=[text, voice, speed],
        outputs=[preview, download],
        concurrency_limit=1,
    )

# Replace the two launch calls with this single line:
app.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
)

# app.queue().launch(
#     server_name="127.0.0.1",
#     server_port=7860,
#     share=False,
#     inbrowser=True,
# )

# app.launch(server_name="0.0.0.0", server_port=7860)
