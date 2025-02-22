import os
import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset, Audio
import evaluate
import argparse
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path

class LoggerSetup:
    @staticmethod
    def setup(log_file: str = 'whisper_training.log') -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

logger = LoggerSetup.setup()

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for batching speech recognition data."""
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty feature list provided")

        try:
            # Process input features
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Process labels
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding with -100 for loss calculation
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # Ensure proper start token
            labels = labels.masked_fill(
                labels.eq(self.processor.tokenizer.pad_token_id),
                self.decoder_start_token_id
            )

            return {
                "input_features": batch["input_features"],
                "labels": labels,
                "attention_mask": batch["attention_mask"]
            }
        except Exception as e:
            logger.error(f"Data collation failed: {str(e)}")
            raise

class AudioDataProcessor:
    """Handles audio data processing and dataset preparation."""
    
    def __init__(
        self, 
        csv_path: str, 
        processor: WhisperProcessor,
        sample_rate: int = 16000,
        max_duration: float = 30.0
    ):
        self.validate_file_exists(csv_path)
        self.data = pd.read_csv(csv_path)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration

    @staticmethod
    def validate_file_exists(file_path: str) -> None:
        """Validate if file exists."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def process_audio_file(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and validate audio file."""
        waveform, orig_sr = torchaudio.load(audio_path)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, orig_sr

    def resample_audio(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate if needed."""
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            )
            return resampler(waveform)
        return waveform

    def validate_audio_duration(self, waveform: torch.Tensor, sr: int) -> bool:
        """Check if audio duration is within limit."""
        duration = waveform.shape[1] / sr
        return duration <= self.max_duration

    def prepare_dataset(self) -> Dataset:
        """Prepare dataset from audio files and transcripts."""
        valid_examples = []
        
        for idx, row in self.data.iterrows():
            try:
                # Load and process audio
                waveform, orig_sr = self.process_audio_file(row['audio_path'])
                
                # Validate duration
                if not self.validate_audio_duration(waveform, orig_sr):
                    logger.warning(f"Skipping {row['audio_path']}: Duration exceeds {self.max_duration}s")
                    continue
                
                # Resample if needed
                waveform = self.resample_audio(waveform, orig_sr)
                
                # Process through Whisper
                inputs = self.processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                
                # Create example
                example = {
                    "input_features": inputs.input_features.squeeze().numpy(),
                    "labels": self.processor.tokenizer.encode(
                        row['transcript'],
                        return_tensors='pt'
                    ).squeeze().numpy()
                }
                
                valid_examples.append(example)
                
            except Exception as e:
                logger.error(f"Error processing example {idx}: {str(e)}")
                continue
                
        if not valid_examples:
            raise ValueError("No valid examples found in the dataset")
            
        return Dataset.from_list(valid_examples)

class WhisperTrainer:
    """Manages the training process for Whisper model."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.processor = None
        self.model = None
        self.metric = evaluate.load("wer")

    def setup_model(self):
        """Initialize model and processor."""
        logger.info(f"Loading model and processor from {self.args.model_name}")
        self.processor = WhisperProcessor.from_pretrained(self.args.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.args.model_name)
        
        # Freeze encoder if specified
        if self.args.freeze_encoder:
            logger.info("Freezing encoder parameters")
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False

    def compute_metrics(self, pred):
        """Compute WER metric."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 padding
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode predictions and references
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def get_training_args(self) -> Seq2SeqTrainingArguments:
        """Configure training arguments."""
        return Seq2SeqTrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            max_steps=self.args.max_steps,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=self.args.save_steps,
            eval_steps=self.args.eval_steps,
            logging_steps=self.args.logging_steps,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False
        )

    def train(self):
        """Execute training process."""
        try:
            # Setup
            self.setup_model()
            
            # Prepare datasets
            logger.info("Preparing datasets...")
            train_processor = AudioDataProcessor(self.args.train_csv, self.processor)
            train_dataset = train_processor.prepare_dataset()
            
            eval_processor = AudioDataProcessor(self.args.val_csv, self.processor)
            eval_dataset = eval_processor.prepare_dataset()

            # Setup data collator
            data_collator = DataCollatorSpeechSeq2SeqWithPadding(
                processor=self.processor,
                decoder_start_token_id=self.model.config.decoder_start_token_id
            )

            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=self.get_training_args(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )

            # Train
            logger.info("Starting training...")
            trainer.train()
            
            # Save final model
            logger.info(f"Saving model to {self.args.output_dir}")
            trainer.save_model(self.args.output_dir)
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning")
    
    # Data arguments
    parser.add_argument('--train_csv', required=True, help='Path to training CSV')
    parser.add_argument('--val_csv', required=True, help='Path to validation CSV')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    # Model arguments
    parser.add_argument('--model_name', default='openai/whisper-large-v2', 
                       help='Model name or path')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder parameters')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps for gradient accumulation')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=4000,
                       help='Maximum number of training steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Save checkpoint every X steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                       help='Evaluate every X steps')
    parser.add_argument('--logging_steps', type=int, default=200,
                       help='Log every X steps')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize trainer and start training
        trainer = WhisperTrainer(args)
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
