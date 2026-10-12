import os
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

VOICES = {
    "Heart — female": "af_heart",
    "Bella — female": "af_bella",
    "Michael — male": "am_michael",
    "Puck — male": "am_puck",
}

OUTPUT = Path(__file__).resolve().parent / "recordings"
OUTPUT.mkdir(exist_ok=True)

# Lazy loading to prevent out-of-memory crashes on free cloud tiers
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a", device="cpu")
    return pipeline

def generate(text, voice_name, speed):
    text = text.strip()
    if not text:
        raise gr.Error("Enter the text you want spoken.")

    pipe = get_pipeline()
    chunks = []
    
    for _, _, audio in pipe(
        text, voice=VOICES[voice_name], speed=float(speed)
    ):
        if audio is not None:
            chunks.append(audio.cpu().numpy())

    if not chunks:
        raise gr.Error("No audio was generated.")

    audio = np.concatenate(chunks)
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

# Get port dynamically from Render or fallback to 10000 / 7860
port = int(os.environ.get("PORT", 10000))

app.queue().launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,
)
