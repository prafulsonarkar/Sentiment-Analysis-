# Set UTF-8 locale (for Colab/Kaggle compatibility)
import os
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

# Install dependencies (verify on online PC)
!pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 -q
!pip install transformers==4.37.0 datasets==2.17.0 librosa==0.10.1 jiwer==3.0.3 -q
!pip install accelerate==0.27.2 soundfile==0.12.1 peft==0.8.2 gtts==2.5.1 -q
!apt update && apt install ffmpeg -y -q

import torch
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
import pandas as pd
from gtts import gTTS
from jiwer import wer
import librosa
import soundfile as sf

# Check GPU memory and availability
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Available: Yes, Device: {torch.cuda.current_device()}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("GPU Not Available!")

# Constants
MODEL_NAME = "vasista22/whisper-hindi-large-v2"
LANGUAGE = "hi"
TASK = "transcribe"
SAMPLING_RATE = 16000
OUTPUT_DIR = "/kaggle/working/checkpoints" if os.path.exists("/kaggle/working") else "./checkpoints"
TRAIN_CSV = "/kaggle/working/train.csv" if os.path.exists("/kaggle/working") else "./train.csv"
TEST_CSV = "/kaggle/working/test.csv" if os.path.exists("/kaggle/working") else "./test.csv"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample CSV generation (for online testing; replace with real data offline)
train_sentences = [
    "नमस्ते, मेरा नाम राहुल है।", "आज मौसम बहुत अच्छा है।", "मुझे किताबें पढ़ना पसंद है।",
    "क्या आप हिंदी बोलते हैं?", "यह एक सुंदर फूल है।", "मैं हर दिन सुबह दौड़ता हूँ।",
    "भारत एक बड़ा देश है।", "मुझे चाय पीना पसंद है।", "कृपया मुझे पानी दें।",
    "यह मेरा नया घर है।", "सूरज सुबह उगता है।", "मैं स्कूल जाता हूँ।",
    "यह कितना सुंदर दृश्य है।", "मुझे संगीत सुनना अच्छा लगता है।", "आपका दिन कैसा रहा?",
    "मैं अपने दोस्त से मिलने जा रहा हूँ।", "यह एक पुराना मंदिर है।", "कृपया धीरे बोलें।",
    "मुझे भारतीय खाना पसंद है।", "आज रात चाँद बहुत सुंदर है।",
]
test_sentences = [
    "कल मैं बाजार गया था।", "यह मेरा पसंदीदा गाना है।", "क्या समय हो रहा है?",
    "मैं अपने परिवार के साथ हूँ।", "यह एक लंबी यात्रा थी।",
]

def generate_audio(text, output_path, lang="hi"):
    temp_mp3 = output_path.replace(".wav", ".mp3")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(temp_mp3)
    audio, sr = librosa.load(temp_mp3, sr=SAMPLING_RATE)
    sf.write(output_path, audio, SAMPLING_RATE)
    os.remove(temp_mp3)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to create audio file: {output_path}")

# Generate sample audio (for online testing only)
train_data = []
for i, sentence in enumerate(train_sentences):
    audio_path = f"{os.path.dirname(TRAIN_CSV)}/train_audio_{i+1}.wav"
    generate_audio(sentence, audio_path)
    train_data.append({"audio_path": audio_path, "transcription": sentence})

test_data = []
for i, sentence in enumerate(test_sentences):
    audio_path = f"{os.path.dirname(TEST_CSV)}/test_audio_{i+1}.wav"
    generate_audio(sentence, audio_path)
    test_data.append({"audio_path": audio_path, "transcription": sentence})

pd.DataFrame(train_data).to_csv(TRAIN_CSV, index=False)
pd.DataFrame(test_data).to_csv(TEST_CSV, index=False)
print(f"Generated {TRAIN_CSV} and {TEST_CSV} with sample WAV files (for online testing)")

# Verify audio files and CSV contents
print("Checking audio files:")
!ls {os.path.dirname(TRAIN_CSV)}/train_audio_*.wav
!ls {os.path.dirname(TEST_CSV)}/test_audio_*.wav
print("Train CSV contents:")
print(pd.read_csv(TRAIN_CSV).head())
print("Test CSV contents:")
print(pd.read_csv(TEST_CSV).head())

# Load processor and model
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Enable gradients explicitly for training
model.train()
for param in model.parameters():
    param.requires_grad_(True)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
model.to("cuda" if torch.cuda.is_available() else "cpu")
print_gpu_memory()

# Load datasets
def load_custom_dataset(csv_path):
    dataset = Dataset.from_pandas(pd.read_csv(csv_path))
    print(f"Dataset loaded from {csv_path}, size: {len(dataset)}")
    dataset = dataset.map(lambda x: {"audio": x["audio_path"], "sentence": x["transcription"]})
    print(f"Dataset after mapping columns, size: {len(dataset)}")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    print(f"Dataset after audio casting, size: {len(dataset)}")
    return dataset

train_dataset = load_custom_dataset(TRAIN_CSV)
test_dataset = load_custom_dataset(TEST_CSV)

# Preprocess with error handling
def preprocess_function(examples):
    try:
        audio = examples["audio"]
        if audio["array"] is None or len(audio["array"]) == 0:
            raise ValueError("Audio array is empty or None")
        inputs = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        labels = processor.tokenizer(examples["sentence"]).input_ids
        if inputs is None or labels is None:
            raise ValueError("Preprocessing returned None for inputs or labels")
        return {"input_features": inputs, "labels": labels}
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

print("Preprocessing train dataset...")
train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio", "sentence"], desc="Mapping train data")
print(f"Train dataset after preprocessing, size: {len(train_dataset)}")
if len(train_dataset) == 0:
    raise ValueError("Train dataset is empty after preprocessing!")

print("Preprocessing test dataset...")
test_dataset = test_dataset.map(preprocess_function, remove_columns=["audio", "sentence"], desc="Mapping test data")
print(f"Test dataset after preprocessing, size: {len(test_dataset)}")
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after preprocessing!")

# Data collator
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor): self.processor = processor
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# WER computation with detailed debugging
def compute_metrics(pred):
    print("Entering compute_metrics...")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    print(f"Predictions: {pred_str}")
    print(f"Ground Truth: {label_str}")
    wer_score = wer(label_str, pred_str)
    print(f"Computed WER: {wer_score}")
    return {"eval_wer": wer_score}

# Training arguments with explicit checkpointing settings
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=10,  # Small for online test; increase offline
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=5,
    save_steps=5,
    save_total_limit=2,
    logging_steps=2,
    logging_strategy="steps",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Verify before training
print(f"Final check - Train dataset size: {len(train_dataset)}")
print(f"Final check - Test dataset size: {len(test_dataset)}")
print_gpu_memory()

# Pre-training evaluation to confirm metrics
print("Running pre-training evaluation...")
eval_results = trainer.evaluate()
print(f"Pre-training WER: {eval_results['eval_wer']:.4f}")

# Train and evaluate
print("Starting training...")
trainer.train()
print("Training completed. Starting evaluation...")
eval_results = trainer.evaluate()
print(f"Final WER: {eval_results['eval_wer']:.4f}")
print_gpu_memory()

# Save model
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))