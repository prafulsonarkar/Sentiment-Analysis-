import torch
import pandas as pd
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

class WhisperFineTuner:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        output_dir: str = "whisper_finetuned",
        sample_rate: int = 16000,
        mixed_precision: str = "fp16",
        max_steps: int = 4000,
        eval_steps: int = 500,
        learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 2,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 8,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        
        # Training configuration
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            max_steps=max_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_steps=eval_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            fp16=mixed_precision == "fp16",
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            logging_dir=f"{str(self.output_dir)}/logs",
            logging_steps=100,
            save_total_limit=3,
        )

        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            load_in_8bit=False,
            device_map="auto"
        )
        
        # Setup metrics
        self.metric = evaluate.load("wer")

    def prepare_dataset(self, csv_path: str, split_name: str = "train") -> Dataset:
        """
        Prepare dataset from CSV file containing audio paths and transcriptions
        Expected columns: audio_path, transcription
        """
        logger.info(f"Loading {split_name} dataset from {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Verify required columns
        required_columns = ["audio_path", "transcription"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Convert DataFrame to Dataset
        dataset = Dataset.from_pandas(df)
        
        def process_data(batch):
            # Load and resample audio
            audio = batch["audio"]
            
            # Extract features
            batch["input_features"] = self.processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            ).input_features[0]

            # Tokenize text
            batch["labels"] = self.processor.tokenizer(batch["transcription"]).input_ids
            return batch

        # Add audio loading capability
        dataset = dataset.cast_column(
            "audio_path", 
            Audio(sampling_rate=self.sample_rate)
        )
        dataset = dataset.rename_column("audio_path", "audio")
        
        # Process dataset
        processed_dataset = dataset.map(
            process_data,
            remove_columns=dataset.column_names,
            num_proc=4
        )

        return processed_dataset

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and references
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def train(self, train_csv: str, eval_csv: str):
        """
        Fine-tune the model using CSV datasets
        """
        logger.info("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_csv, "train")
        eval_dataset = self.prepare_dataset(eval_csv, "eval")

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor),
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        logger.info("Starting training...")
        trainer.train()

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.processor.save_pretrained(str(self.output_dir / "final_model"))

def main():
    # Initialize fine-tuner
    fine_tuner = WhisperFineTuner(
        model_name="openai/whisper-large-v2",
        output_dir="whisper_finetuned_multilingual",
        max_steps=4000,
        eval_steps=500,
        learning_rate=1e-5,
        warmup_steps=500,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
    )

    # CSV file paths
    train_csv = "path/to/train.csv"
    eval_csv = "path/to/eval.csv"

    # Start training
    fine_tuner.train(train_csv, eval_csv)

if __name__ == "__main__":
    main()
