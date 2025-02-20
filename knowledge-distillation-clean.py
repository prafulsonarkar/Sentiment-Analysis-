import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import jiwer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('asr_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CompactStudentModel(nn.Module):
    """Compact student model for ASR knowledge distillation"""
    def __init__(
        self, 
        input_dim: int = 80,
        hidden_dim: int = 512, 
        num_heads: int = 8,
        num_layers: int = 4,
        vocab_size: int = 50257,  # Default Whisper vocab size
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else hidden_dim, 
                         hidden_dim, 
                         kernel_size=3, 
                         padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(4)
        ])
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output
        self.final_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1500, hidden_dim)  # Max sequence length of 1500
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        tgt: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x shape: [batch_size, time_steps, features]
        batch_size, time_steps, _ = x.shape
        
        # Convolutional encoding
        x = x.transpose(1, 2)  # [batch_size, features, time_steps]
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.transpose(1, 2)  # [batch_size, time_steps, hidden_dim]
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :time_steps, :]
        
        # Transformer encoding
        memory = self.transformer_encoder(x)
        
        # Decoder
        if tgt is None:
            # For inference, use memory as target
            tgt = memory
        
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask
        )
        
        # Final projection
        output = self.final_layer(output)
        
        return output

class MultilingualASRDataset(Dataset):
    def __init__(self, csv_path: str, processor: WhisperProcessor):
        self.data = pd.read_csv(csv_path)
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            # Load and process audio
            waveform, sample_rate = torchaudio.load(row['audio_path'])
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)
            
            # Process audio with Whisper processor
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Process transcript
            labels = self.processor.tokenizer(
                row['transcript'],
                return_tensors="pt"
            ).input_ids.squeeze()
            
            return {
                'input_features': inputs.input_features.squeeze(),
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Error processing {row['audio_path']}: {str(e)}")
            # Return None or raise exception based on your error handling strategy
            raise

class KnowledgeDistillationTrainer:
    def __init__(
        self,
        teacher_model_name: str,
        student_config: Dict,
        training_config: Dict,
        device