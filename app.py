import gradio as gr
from faster_whisper import WhisperModel
from time import time
import logging
import json

# Initialize logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
CHOICES = [
    "tiny", "tiny.en", "base", 
    "base.en", "small", "small.en", 
    "medium", "medium.en", "large-v1", 
    "large-v2", "large-v3", "large"
]
# Function to load model
def load_model(model):
    download_path_int8 = "int8"  # Adjust path as needed for Hugging Face Spaces
    return WhisperModel(model, device="auto", compute_type="int8", download_root=download_path_int8)

# Current model (default to small)
current_model = load_model("small")

def transcribe(audio_file, model):
    global current_model

    # Load the model if different size is selected
    if current_model.model != model:
        current_model = load_model(model)

    start = time()
    segments, info = current_model.transcribe(
        audio_file,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Prepare JSON output
    transcript = [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments]
    print(f"Time Taken to transcribe: {time() - start}")
    output = {
        "language": info.language,
        "language_probability": info.language_probability,
        "transcript": transcript
    }

    return json.dumps(output, indent=4)

# Define Gradio interface
iface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath", label="Upload MP3 Audio File"),
        gr.Dropdown(choices=CHOICES, value="small", label="Model")
    ],
    outputs=gr.JSON(label="Transcription with Timestamps"),
    title="Whisper Transcription Service",
    description="Upload an MP3 audio file to transcribe. Select the model. The output includes the transcription with timestamps."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
