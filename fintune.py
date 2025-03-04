import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import evaluate

# Load dataset
def load_custom_dataset(csv_path):
    dataset = load_dataset("csv", data_files={"train": csv_path})
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    return dataset

# Preprocess dataset
def prepare_dataset(dataset, processor):
    def process_example(batch):
        audio = batch["audio_path"]["array"]
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        labels = processor.tokenizer(batch["transcript"]).input_ids
        return {"input_features": input_features, "labels": labels}

    dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    return dataset

# Custom callback to save the best model
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.best_wer = float("inf")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics["eval_wer"] < self.best_wer:
            self.best_wer = metrics["eval_wer"]
            model.save_pretrained(os.path.join(self.output_dir, "best_model"))
            processor.save_pretrained(os.path.join(self.output_dir, "best_model"))

# Load processor and model from local directory
model_name = "./whisper-small-hindi"  # Path to the locally saved model
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Set language and task for model
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")

# Load dataset
dataset = load_custom_dataset("data.csv")
dataset = prepare_dataset(dataset, processor)

# Split dataset into train and validation sets
dataset = dataset["train"].train_test_split(test_size=0.1)

# Define data collator
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(processor, model=model)

# Define evaluation metric
metric = evaluate.load("wer")  # Word Error Rate

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token ID
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned-hindi",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    fp16=True,  # Use mixed precision if you have a GPU
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=100,
    report_to=["tensorboard"],
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    gradient_clipping=1.0,  # Prevent exploding gradients
    weight_decay=0.01,  # Regularization to prevent overfitting
    lr_scheduler_type="cosine",  # Cosine annealing with warm restarts
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestModelCallback("./whisper-finetuned-hindi")],  # Save the best model
)

# Start training
trainer.train()

# Save the final model
trainer.save_model("./whisper-finetuned-hindi/final_model")
processor.save_pretrained("./whisper-finetuned-hindi/final_model")