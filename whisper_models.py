https://www.kaggle.com/code/prafulsonarkar/whisper-hindi-v2-large
# Step 2: Import libraries
import librosa
import soundfile as sf
from google.colab import files


input_audio_path = "/kaggle/input/smallest-audio-dataset/hindi_audio6.mp3"


# Step 4: Load the audio file
y, sr = librosa.load(input_audio_path, sr=24000)  # Load audio at 24,000 Hz

# Step 5: Resample the audio to 16,000 Hz
y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)

# Step 6: Save the resampled audio
output_audio_path = "resampled_audio.wav"
sf.write(output_audio_path, y_resampled, 16000)

# Step 7: Download the resampled audio
files.download(output_audio_path)

print(f"Resampled audio saved to: {output_audio_path}")


import torch
from transformers import pipeline

# Check if CUDA is available for GPU processing
device = 0 if torch.cuda.is_available() else -1

# Load the Whisper model and tokenizer for Hindi transcription
transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v2",
    chunk_length_s=30,
    device=device
)

# Optional: Set forced decoder ids for Hindi (language code 'hi')
transcriber.model.config.forced_decoder_ids = transcriber.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

# Path to your Hindi audio file
audio_file = "/kaggle/input/smallest-audio-dataset/hindi_audio6.mp3"

# Transcribe the audio file
result = transcriber(audio_file)

# Print the transcription result
print("Transcription (Hindi): ", result["text"])




#import torch
#from transformers import pipeline
audio = "/kaggle/input/smallest-audio-dataset/hindi_audio7.mp3"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-hindi-small", chunk_length_s=30, device=device)

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

result = transcriber(audio)

print('Transcription: ', result["text"])



!pip install git+https://github.com/huggingface/transformers gradio


import torch
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v3-turbo",
               torch_dtype=torch.float16,
               device="cuda:0")

pipe("/kaggle/input/smallest-audio-dataset/hindi_audio7.mp3")




