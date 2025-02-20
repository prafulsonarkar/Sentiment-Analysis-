import os
import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whisper_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        # Validate input
        if not features:
            raise ValueError("Empty feature list provided")

        try:
            # Preprocess audio
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Preprocess labels
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding with -100 to ignore loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Ensure labels start with decoder start token
            labels = labels.masked_fill(labels.eq(self.processor.tokenizer.pad_token_id), self.decoder_start_token_id)

            return {
                "input_features": batch["input_features"],
                "labels": labels,
                "attention_mask": batch["attention_mask"]
            }
        except Exception as e:
            logger.error(f"Error in data collation: {str(e)}")
            raise

class AudioDataProcessor:
    def __init__(
        self, 
        csv_path: str, 
        processor: WhisperProcessor,
        sample_rate: int = 16000,
        max_duration: float = 30.0
    ):
        """
        Initialize audio data processor.
        
        Args:
            csv_path: Path to CSV file containing audio paths and transcripts
            processor: WhisperProcessor instance
            sample_rate: Target sample rate for audio
            max_duration: Maximum allowed duration for audio files in seconds
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.data = pd.read_csv(csv_path)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
    
    def validate_audio_duration(self, waveform: torch.Tensor, sr: int) -> bool:
        """Validate if audio duration is within allowed limit."""
        duration = waveform.shape[1] / sr
        return duration <= self.max_duration

    def prepare_dataset(self):
        def process_audio(example: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
            try:
                # Load Audio
                waveform, orig_sr = torchaudio.load(example['audio_path'])
                
                # Ensure 2D tensor
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                
                # Validate duration
                if not self.validate_audio_duration(waveform, orig_sr):
                    logger.warning(f"Audio file too long: {example['audio_path']}")
                    return None
                
                # Resample if needed
                if orig_sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=orig_sr,
                        new_freq=self.sample_rate
                    )
                    waveform = resampler(waveform)
                
                # Whisper Processing
                inputs = self.processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                
                return {
                    "input_features": inputs.input_features.squeeze().numpy(),
                    "labels": self.processor.tokenizer.encode(
                        example['transcript'],
                        return_tensors='pt'
                    ).squeeze().numpy()
                }
            except Exception as e:
                logger.error(f"Error processing {example['audio_path']}: {str(e)}")
                return None
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(self.data)
        processed_dataset = dataset.map(
            process_audio,
            remove_columns=dataset.column_names,
            filter_by_format=True
        )
        
        # Filter out None values
        processed_dataset = processed_dataset.filter(
            lambda x: x is not None and all(v is not None for v in x.values())
        )
        
        return processed_dataset

def main():
    parser = argparse.ArgumentParser(description="Whisper Hindi Fine-Tuning")
    parser.add_argument('--train_csv', required=True, help='Path to training CSV')
    parser.add_argument('--val_csv', required=True, help='Path to validation CSV')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--model_name', default='vasista/whisper-hindi-large-v2', help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')
    args = parser.parse_args()

    try:
        # Model and Processor Initialization
        logger.info(f"Loading model and processor from {args.model_name}")
        processor = WhisperProcessor.from_pretrained(args.model_name)
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
        
        # Freeze Encoder
        for param in model.model.encoder.parameters():
            param.requires_grad = False

        # Data Preparation
        logger.info("Preparing datasets...")
        data_processor = AudioDataProcessor(args.train_csv, processor)
        train_dataset = data_processor.prepare_dataset()
        
        data_processor_val = AudioDataProcessor(args.val_csv, processor)
        eval_dataset = data_processor_val.prepare_dataset()

        # Metric
        metric = evaluate.load("wer")
        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            
            # Replace -100
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
            
            wer = metric.compute(predictions=pred_str, references=label_str)
            return {"wer": wer}

        # Data Collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )

        # Training Arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            warmup_steps=500,
            max_steps=args.max_steps,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=200,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Start Training
        logger.info("Starting training...")
        trainer.train()
        
        # Save Final Model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
