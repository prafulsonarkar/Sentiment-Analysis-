import os
import torch
import argparse
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
def load_custom_dataset(train_csv_path, test_csv_path):
    train_dataset = load_dataset("csv", data_files={"train": train_csv_path})
    test_dataset = load_dataset("csv", data_files={"test": test_csv_path})
    
    # Cast audio column to Audio type
    train_dataset = train_dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    
    return train_dataset["train"], test_dataset["test"]

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
            student_model.save_pretrained(os.path.join(self.output_dir, "best_model"))
            processor.save_pretrained(os.path.join(self.output_dir, "best_model"))

# Load processor and model from local directory
def load_teacher_model(teacher_model_dir):
    teacher_processor = WhisperProcessor.from_pretrained(teacher_model_dir)
    teacher_model = WhisperForConditionalGeneration.from_pretrained(teacher_model_dir)
    return teacher_processor, teacher_model

def load_student_model(student_model_dir):
    student_processor = WhisperProcessor.from_pretrained(student_model_dir)
    student_model = WhisperForConditionalGeneration.from_pretrained(student_model_dir)
    return student_processor, student_model

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

def main(train_csv_path, test_csv_path, teacher_model_dir, student_model_dir, output_dir):
    # Load teacher and student models
    teacher_processor, teacher_model = load_teacher_model(teacher_model_dir)
    student_processor, student_model = load_student_model(student_model_dir)

    # Set language and task for models
    teacher_model.config.forced_decoder_ids = teacher_processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    student_model.config.forced_decoder_ids = student_processor.get_decoder_prompt_ids(language="hi", task="transcribe")

    # Load dataset
    train_dataset, test_dataset = load_custom_dataset(train_csv_path, test_csv_path)
    train_dataset = prepare_dataset(train_dataset, teacher_processor)
    test_dataset = prepare_dataset(test_dataset, teacher_processor)

    # Define data collator
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(teacher_processor, model=teacher_model)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # Adjust based on GPU memory
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,  # Lower learning rate for distillation
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

    # Define trainer for knowledge distillation
    class DistillationTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs["input_features"])

            # Get student predictions
            student_outputs = model(inputs["input_features"])

            # Compute distillation loss (e.g., KL divergence)
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_outputs.logits, dim=-1),
                torch.nn.functional.softmax(teacher_outputs.logits, dim=-1),
                reduction="batchmean",
            )

            if return_outputs:
                return loss, student_outputs
            return loss

    # Define trainer
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=teacher_processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SaveBestModelCallback(output_dir)],  # Save the best model
    )

    # Start training
    trainer.train()

    # Save the final student model
    trainer.save_model(os.path.join(output_dir, "final_model"))
    student_processor.save_pretrained(os.path.join(output_dir, "final_model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation for ASR")
    parser.add_argument("--train_csv", required=True, help="Path to the training CSV file")
    parser.add_argument("--test_csv", required=True, help="Path to the testing CSV file")
    parser.add_argument("--teacher_model_dir", required=True, help="Path to the teacher model directory")
    parser.add_argument("--student_model_dir", required=True, help="Path to the student model directory")
    parser.add_argument("--output_dir", required=True, help="Path to save the fine-tuned student model")
    args = parser.parse_args()

    # Run the main function
    main(args.train_csv, args.test_csv, args.teacher_model_dir, args.student_model_dir, args.output_dir)