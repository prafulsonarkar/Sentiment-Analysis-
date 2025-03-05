import pandas as pd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio
from torch.utils.data import DataLoader
import torchaudio
import evaluate

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV
df = pd.read_csv("your_data.csv")  # Columns: audio_path, transcription
train_df = df.sample(frac=0.8, random_state=42)  # 80% train
test_df = df.drop(train_df.index)  # 20% test

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df).cast_column("audio_path", Audio(sampling_rate=16000))
test_dataset = Dataset.from_pandas(test_df).cast_column("audio_path", Audio(sampling_rate=16000))

# Load processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Preprocess function
def preprocess(batch):
    audio = batch["audio_path"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"], return_tensors="pt").input_ids[0]
    return batch

train_dataset = train_dataset.map(preprocess, remove_columns=["audio_path", "transcription"])
test_dataset = test_dataset.map(preprocess, remove_columns=["audio_path", "transcription"])

# Load teacher and student models
teacher_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)
student_model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3").to(device)

# Fine-tune teacher (optional but recommended)
teacher_model.train()
optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=1e-5)
for epoch in range(3):  # Adjust epochs
    for batch in DataLoader(train_dataset, batch_size=4, shuffle=True):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        outputs = teacher_model(input_features, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Teacher Epoch {epoch+1}, Loss: {loss.item()}")

# Knowledge distillation
student_model.train()
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
temperature = 2.0  # For softening teacher logits
alpha = 0.7  # Weight for distillation loss
wer_metric = evaluate.load("wer")

for epoch in range(5):  # Adjust epochs
    total_loss = 0
    for batch in DataLoader(train_dataset, batch_size=4, shuffle=True):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        # Teacher predictions (soft labels)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_features)
            teacher_logits = teacher_outputs.logits / temperature

        # Student predictions
        student_outputs = student_model(input_features, labels=labels)
        student_logits = student_outputs.logits / temperature

        # Distillation loss (KL-divergence)
        distill_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)

        # CTC loss with ground truth
        ctc_loss = student_outputs.loss

        # Combined loss
        loss = alpha * distill_loss + (1 - alpha) * ctc_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Student Epoch {epoch+1}, Loss: {total_loss / len(train_dataset)}")

# Evaluation
student_model.eval()
predictions, references = [], []
for batch in DataLoader(test_dataset, batch_size=4):
    input_features = batch["input_features"].to(device)
    with torch.no_grad():
        generated_ids = student_model.generate(input_features)
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    ref_text = processor.batch_decode(batch["labels"], skip_special_tokens=True)
    predictions.extend(pred_text)
    references.extend(ref_text)

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"Test WER: {wer}")

# Save student model
student_model.save_pretrained("distilled_student_model")
processor.save_pretrained("distilled_student_model")