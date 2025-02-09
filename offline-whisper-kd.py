# multilingual_whisper_kd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm
import jiwer
from typing import Dict, Optional, List

# Setup logging with more detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rest of the classes remain the same until MultilingualTrainer

class MultilingualTrainer:
    """Trainer for multilingual knowledge distillation"""
    def __init__(
        self,
        teacher_model_path: str,  # Changed to accept local path
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        temperature: float = 2.5,
        distillation_alpha: float = 0.6,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.distillation_alpha = distillation_alpha
        
        # Initialize models from local path
        logger.info(f"Loading teacher model from {teacher_model_path}")
        self.processor = WhisperProcessor.from_pretrained(teacher_model_path, local_files_only=True)
        self.teacher = WhisperForConditionalGeneration.from_pretrained(teacher_model_path, local_files_only=True)
        self.teacher.eval()
        self.teacher.to(self.device)
        
        self.student = CompactStudentModel(
            vocab_size=self.processor.tokenizer.vocab_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Training components
        self.optimizer = optim.AdamW(self.student.parameters(), lr=learning_rate)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

def main():
    # Configuration
    config = {
        'train_csv': 'data/train.csv',  # Update with your CSV path
        'val_csv': 'data/val.csv',      # Update with your CSV path
        'save_dir': 'checkpoints',
        'teacher_model_path': 'path/to/local/whisper-hindi-large-v2',  # Update with your local model path
        'epochs': 50,
        'batch_size': 16,
        'hidden_dim': 512,
        'learning_rate': 1e-4,
        'temperature': 2.5,
        'distillation_alpha': 0.6,
        'checkpoint_path': None  # Set to checkpoint path if resuming training
    }
    
    # Create save directory
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = MultilingualTrainer(
        teacher_model_path=config['teacher_model_path'],
        hidden_dim=config['hidden_dim'],
        learning_rate=config['learning_rate'],
        temperature=config['temperature'],
        distillation_alpha=config['distillation_alpha']
    )
    
    # Create dataloaders
    train_dataset = AudioDataset(config['train_csv'], trainer.processor)
    val_dataset = AudioDataset(config['val_csv'], trainer.processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train model
    trainer.train(
        train_loader,
        val_loader,
        config['epochs'],
        config['save_dir'],
        config['checkpoint_path']
    )

if __name__ == "__main__":
    main()
