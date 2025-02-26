# Install PyTorch with CUDA 12.1 support (adjust CUDA version based on your GPU)
!pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 -q

# Install core libraries for fine-tuning Whisper
!pip install transformers==4.37.0 datasets==2.17.0 librosa==0.10.1 jiwer==3.0.3 -q

# Install additional tools for training and audio processing
!pip install accelerate==0.27.2 soundfile==0.12.1 peft==0.8.2 -q

# Install ffmpeg for audio preprocessing (Ubuntu-specific)
!apt update && apt install ffmpeg -y -q
# Set UTF-8 locale (ensure compatibility on Ubuntu)
import os
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

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

# Constants (update these paths as per your offline GPU PC)
MODEL_NAME = "vasista22/whisper-hindi-large-v2"  # Update to local path if pre-downloaded
LANGUAGE = "hi"
TASK = "transcribe"
SAMPLING_RATE = 16000
OUTPUT_DIR = "./checkpoints"  # Path on your GPU PC
TRAIN_CSV = "./train.csv"     # Path to your train.csv
TEST_CSV = "./test.csv"       # Path to your test.csv

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify CSV contents
print("Train CSV contents:")
print(pd.read_csv(TRAIN_CSV).head())
print("Test CSV contents:")
print(pd.read_csv(TEST_CSV).head())

# Load processor and model (use local path if offline)
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
            raise ValueError(f"Audio array is empty or None for {examples['audio']['path']}")
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
        labels_batch = self.processor.tokenizer.pad(label_feature
s, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# WER computation with debugging
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

# Training arguments tailored for your dataset
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,      # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch size = 16
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=10000,                    # Suitable for 522 hours of data
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=True,                          # Mixed precision for GPU
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=3,                 # Keep last 3 checkpoints
    logging_steps=50,
    logging_strategy="steps",
    report_to="none",                   # No online reporting (offline)
    load_best_model_at_end=True,
    metric_for_best_model="eval_wer",
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

# Pre-training evaluation
print("Running pre-training evaluation...")
eval_results = trainer.evaluate()
print(f"Pre-training WER: {eval_results['eval_wer']:.4f}")

# Train and evaluate
print("Starting training...")
trainer.train()
print("Training completed. Starting final evaluation...")
eval_results = trainer.evaluate()
print(f"Final WER: {eval_results['eval_wer']:.4f}")
print_gpu_memory()

# Save final model
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print(f"Model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")