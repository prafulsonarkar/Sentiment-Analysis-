# Create environment
conda create -n whisper_env python=3.9
conda activate whisper_env

# Install dependencies
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch -c nvidia
conda install librosa jiwer pandas
pip install git+https://github.com/openai/whisper.git

# Test
python -c "import torch; print(torch.cuda.is_available())"
python -c "import whisper; import pandas; print('Setup OK')"

# Download model
python -c "import whisper; model = whisper.load_model('large-v3'); model.save('whisper_large_v3.pt')"

# Export environment
conda pack -n whisper_env -o whisper_env.tar.gz
import torchaudio
import librosa
import os
import json
import pandas as pd
import whisper
from jiwer import wer

# Step 1: Convert CSV to JSON with Duration
def csv_to_json(csv_file, json_file, audio_dir="audio"):
    df = pd.read_csv(csv_file)
    data = []
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row["audio_path"])  # Adjust column name if different
        duration = librosa.get_duration(filename=audio_path)
        data.append({
            "audio_filepath": audio_path,
            "text": row["transcription"],  # Adjust column name if different
            "duration": duration
        })
    with open(json_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

# Convert your CSV files
os.makedirs("audio_clean", exist_ok=True)
csv_to_json("train.csv", "train.json")
csv_to_json("test.csv", "test.json")

# Step 2: Clean Audio
def preprocess_audio(input_path, output_path):
    waveform, sr = torchaudio.load(input_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.numpy()[0]
    mask = librosa.amplitude_to_db(audio_np) > -40.0
    torchaudio.save(output_path, torch.tensor([audio_np * mask]), 16000)

with open("train.json", 'r') as f:
    train_data = [json.loads(line) for line in f]
for entry in train_data:
    input_path = entry["audio_filepath"]
    output_path = os.path.join("audio_clean", os.path.basename(input_path))
    preprocess_audio(input_path, output_path)
    entry["audio_filepath"] = output_path
with open("train_cleaned.json", 'w') as f:
    for entry in train_data:
        f.write(json.dumps(entry) + '\n')

# Step 3 & 4: Pseudo-Labeling with Prompting
model = whisper.load_model("whisper_large_v3.pt")  # Pre-downloaded model
prompt = "Hindi Urdu Punjabi noisy speech, namaste, shukriya, sat sri akaal, ji, haan"

def generate_pseudo_labels(json_file, output_file):
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]
    new_data = []
    for entry in data:
        result = model.transcribe(entry["audio_filepath"], prompt=prompt, language="hi")
        new_data.append({"audio_filepath": entry["audio_filepath"], "text": result["text"], "duration": entry["duration"]})
    with open(output_file, 'w') as f:
        for entry in new_data:
            f.write(json.dumps(entry) + '\n')

generate_pseudo_labels("train_cleaned.json", "train_pseudo.json")
generate_pseudo_labels("test.json", "test_pseudo.json")

# Step 5: Normalization
def normalize(text):
    return text.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "").replace("namastey", "namaste").replace("shukria", "shukriya")

def normalize_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    for entry in data:
        entry["text"] = normalize(entry["text"])
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

normalize_json("train_pseudo.json", "train_norm.json")
normalize_json("test_pseudo.json", "test_norm.json")

# Step 6: Fine-Tuning
from whisper.finetune import finetune
finetune(
    model,
    train_dataset="train_norm.json",
    val_dataset="test_norm.json",
    epochs=10,
    batch_size=8,
    learning_rate=1e-5,
    output_dir="finetuned_model"
)

# Step 7: Refinement
fine_model = whisper.load_model("finetuned_model/checkpoint-latest")
generate_pseudo_labels("train_cleaned.json", "train_pseudo_v2.json")
normalize_json("train_pseudo_v2.json", "train_norm_v2.json")

# Step 8: Test with WER
def evaluate_model(model_path, test_file):
    model = whisper.load_model(model_path)
    with open(test_file, 'r') as f:
        data = [json.loads(line) for line in f]
    ground_truths = [entry["text"] for entry in data]
    transcriptions = [normalize(model.transcribe(d["audio_filepath"], prompt=prompt, language="hi")["text"]) for d in data]
    print(f"WER: {wer(ground_truths, transcriptions):.4f}")
    # Test new audio
    new_result = model.transcribe("new_noisy.wav", prompt=prompt, language="hi")
    print("New Audio:", normalize(new_result["text"]))

evaluate_model("finetuned_model/checkpoint-latest", "test_norm.json")