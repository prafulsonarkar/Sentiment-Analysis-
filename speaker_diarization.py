import whisper
from pyannote.audio import Pipeline
from aeneas.tools.execute_task import ExecuteTask
from aeneas.task import Task
import noisereduce as nr
import numpy as np
import gradio as gr
import soundfile as sf
import json
import os

# Step 0: Noise reduction
def reduce_noise(audio_file, output_file="cleaned_audio.wav"):
    # Load audio file
    data, rate = sf.read(audio_file)
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # Save cleaned audio
    sf.write(output_file, reduced_noise, rate)
    return output_file

# Step 1: Transcribe audio using Whisper
def transcribe_audio(audio_file, language="hi", model_size="base", progress=gr.Progress()):
    progress(0.1, desc=f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)  # Load the selected model
    progress(0.3, desc="Transcribing audio...")
    result = model.transcribe(audio_file, language=language)
    progress(0.7, desc="Transcription completed.")
    return result["segments"]

# Step 2: Perform speaker diarization using pyannote.audio
def diarize_audio(audio_file, progress=gr.Progress()):
    progress(0.1, desc="Loading pyannote.audio model...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    progress(0.4, desc="Diarizing audio...")
    diarization = pipeline(audio_file)
    progress(0.8, desc="Diarization completed.")
    return diarization

# Step 3: Align words with speakers using aeneas
def align_words_with_speakers(transcription, diarization, audio_file, progress=gr.Progress()):
    # Convert diarization to a list of (start, end, speaker) segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    # Prepare aeneas task for word-level alignment
    task = Task()
    task.config_string = "task_language=hin|is_text_type=plain|os_task_file_format=json"
    task.audio_file_path_absolute = audio_file
    task.text_file_path_absolute = "transcription.txt"
    task.sync_map_file_path_absolute = "sync_map.json"

    # Save transcription to a text file
    with open("transcription.txt", "w") as f:
        for segment in transcription:
            f.write(segment["text"] + "\n")

    # Execute alignment
    progress(0.1, desc="Aligning words with speakers...")
    ExecuteTask(task).execute()

    # Load alignment results
    with open("sync_map.json", "r") as f:
        sync_map = json.load(f)

    # Map words to speakers
    speaker_words = {}
    for word in sync_map["fragments"]:
        word_start = float(word["begin"])
        word_end = float(word["end"])
        word_text = word["lines"][0]

        # Find which speaker segment the word belongs to
        for seg_start, seg_end, speaker in segments:
            if seg_start <= word_start <= seg_end:
                if speaker not in speaker_words:
                    speaker_words[speaker] = []
                speaker_words[speaker].append(word_text)
                break

    progress(0.9, desc="Alignment completed.")
    return speaker_words

# Main function
def process_audio(audio_file, language="hi", model_size="base", progress=gr.Progress()):
    # Step 0: Reduce noise
    progress(0.1, desc="Reducing noise...")
    cleaned_audio = reduce_noise(audio_file)
    print("Noise reduction completed.")

    # Step 1: Transcribe audio
    transcription = transcribe_audio(cleaned_audio, language, model_size, progress)
    print("Transcription completed.")

    # Step 2: Diarize audio
    diarization = diarize_audio(cleaned_audio, progress)
    print("Diarization completed.")

    # Step 3: Align words with speakers
    speaker_words = align_words_with_speakers(transcription, diarization, cleaned_audio, progress)
    result = "\n\n".join([f"Speaker {speaker}: {' '.join(words)}" for speaker, words in speaker_words.items()])

    # Save result to a text file
    with open("result.txt", "w") as f:
        f.write(result)
    print("Result saved to result.txt.")

    return result, "result.txt"

# Gradio GUI
def gradio_interface(audio_file, language, model_size):
    result, file_path = process_audio(audio_file, language, model_size)
    return result, file_path

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(
            choices=["hi", "ur", "pa", "en", "es", "fr", "de", "ja", "zh"],  # Add more languages
            label="Language",
            value="hi"
        ),
        gr.Dropdown(
            choices=["base", "small", "medium", "large"],  # Whisper model sizes
            label="Whisper Model Size",
            value="base"
        )
    ],
    outputs=[
        gr.Textbox(label="Speaker Diarization Result"),
        gr.File(label="Download Result")
    ],
    title="Word-Level Speaker Diarization",
    description="Upload an audio file to perform word-level speaker diarization in multiple languages."
)

# Launch the app
iface.launch()