# -*- coding: utf-8 -*-
"""vasista22_whisper-hindi-large-v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14ook7wZwIEUNkQkyvJ1X1D0ZWXYJkOgW
"""

!pip install transformers=='4.37.0'
!pip install pytorch=='2.1.0'

!pip install torch=='2.1.0'

!python --version

!sudo apt-get update -y
!sudo apt-get install python3.10

!https://huggingface.co/vasista22/whisper-hindi-large-v2

import torch
from transformers import pipeline

# path to the audio file to be transcribed
audio = "/content/resampled_audio.wav"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-hindi-large-v2", chunk_length_s=30, device=device)
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

print('Transcription: ', transcribe(audio)["text"])

!pip install pandas

from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import pandas as pd
from transformers import pipeline

def transcribe_audio_files(input_folder, output_csv):
    """
    Transcribe all .wav and .mp3 files in a folder to a CSV

    Args:
        input_folder (str): Path to folder containing audio files
        output_csv (str): Path to save output CSV file
    """
    # Check CUDA availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize transcription pipeline
    transcribe = pipeline(
        task="automatic-speech-recognition",
        model="vasista22/whisper-hindi-large-v2",
        chunk_length_s=30,
        device=device
    )
    transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

    # Prepare results list
    transcription_results = []

    # Process audio files
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.wav', '.mp3')):
            file_path = os.path.join(input_folder, filename)

            try:
                # Transcribe audio
                transcription = transcribe(file_path)["text"]

                # Store results
                transcription_results.append({
                    'filename': filename,
                    'transcription': transcription
                })

                print(f"Transcribed: {filename}")

            except Exception as e:
                print(f"Error transcribing {filename}: {e}")

    # Save to CSV
    df = pd.DataFrame(transcription_results)
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Transcriptions saved to {output_csv}")

# Example usage
input_folder = "/content/audio"
output_csv = "hindi_transcriptions.csv"
transcribe_audio_files(input_folder, output_csv)

