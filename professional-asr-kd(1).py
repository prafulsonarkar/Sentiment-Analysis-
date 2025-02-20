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

class KnowledgeDistillationTrainer:
    def __init__(
        self,
        teacher_model_name: str,
        student_config: Dict,
        training_config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.config = training_config
        
        # Initialize teacher model
        logger.info(f"Loading teacher model: {teacher_model_name}")
        self.teacher = WhisperForConditionalGeneration.from_pretrained(teacher_model_name)
        self.teacher.to(device)
        self.teacher.eval()
        
        # Initialize processor
        self.processor = WhisperProcessor.from_pretrained(teacher_model_name)
        
        # Initialize student model
        self.student = CompactStudentModel(**student_config)
        self.student.to(device)
        
        # Optimization
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=training_config['scheduler_t0'],
            T_mult=training_config['scheduler_t_mult'],
            eta_min=training_config['scheduler_eta_min']
        )
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Temperature for distillation
        self.temperature = training_config['temperature']
        
        # Loss weights
        self.alpha = training_config['distillation_alpha']  # Weight for distillation loss
        
        # Create output directory
        self.output_dir = Path(training_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.student.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_features = batch['input_features'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = self.teacher(input_features)
            teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = self.student(input_features)
        
        # Calculate losses
        # 1. Distillation loss (KL divergence)
        T = self.temperature
        distillation_loss = self.kl_loss(
            F.log_softmax(student_outputs / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1)
        ) * (T * T)
        
        # 2. Student CE loss with ground truth
        ce_loss = self.ce_loss(
            student_outputs.view(-1, student_outputs.size(-1)),
            labels.view(-1)
        )
        
        # Combined loss
        loss = (self.alpha * distillation_loss + 
                (1 - self.alpha) * ce_loss)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(),
            self.config['max_grad_norm']
        )
        
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'distillation_loss': distillation_loss.item(),
            'ce_loss': ce_loss.item()
        }
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        self.student.eval()
        total_wer = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch in eval_loader:
            input_features = batch['input_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Student predictions
            student_outputs = self.student(input_features)
            
            # Calculate loss
            loss = self.ce_loss(
                student_outputs.view(-1, student_outputs.size(-1)),
                labels.view(-1)
            )
            
            # Calculate WER
            pred_ids = torch.argmax(student_outputs, dim=-1)
            pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
            wer = jiwer.wer(label_str, pred_str)
            
            total_wer += wer
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_wer': total_wer / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_epochs: int
    ):
        best_wer = float('inf')
        step = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                step += 1
                
                # Training step
                train_metrics = self.train_step(batch)
                
                # Logging
                if step % self.config['log_every'] == 0:
                    logger.info(
                        f"Step {step}: loss={train_metrics['total_loss']:.4f}, "
                        f"distill_loss={train_metrics['distillation_loss']:.4f}, "
                        f"ce_loss={train_metrics['ce_loss']:.4f}"
                    )
                
                # Evaluation
                if step % self.config['eval_every'] == 0:
                    eval_metrics = self.evaluate(eval_loader)
                    logger.info(
                        f"Evaluation: loss={eval_metrics['eval_loss']:.4f}, "
                        f"WER={eval_metrics['eval_wer']:.4f}"
                    )
                    
                    # Save best model
                    if eval_metrics['eval_wer'] < best_wer:
                        best_wer = eval_metrics['eval_wer']
                        self.save_checkpoint(
                            f"best_model_step_{step}_wer_{best_wer:.4f}.pt",
                            step,
                            eval_metrics
                        )
                
                # Learning rate scheduling
                self.scheduler.step(epoch + batch_idx / len(train_loader))
    
    def save_checkpoint(
        self,
        filename: str,
        step: int,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="ASR Knowledge Distillation Training")
    parser.add_argument('--train_csv', required=True, help='Path to training CSV')
    parser.add_argument('--val_csv', required=True, help='Path to validation CSV')
    parser.add_argument('--teacher_model', required=True, help='Teacher model name or path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    args = parser.parse_args()

    # Configuration
    student_config = {
        'input_dim': 80,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1
    }
    
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'scheduler_t0': 10,
        'scheduler_t_mult': 2,
        'scheduler_eta_min': 1e-6,
        'temperature': 2.0,
        'distillation_alpha': 0.5,
        'max_grad_norm': 1.0,
        'log_every': 100,
        'eval_every': 1000,
        'output_dir': args.output_dir
    }

    try:
        # Initialize trainer
        trainer = KnowledgeDistillationTrainer(
            teacher_model_name=args.teacher_model,
            student_config=student_config,
            training_config=training_config
        )
        
        # Prepare datasets
        train_dataset = MultilingualASRDataset(
            args.train_csv,
            trainer.processor,
            augment=True
        )
        val_dataset = MultilingualASRDataset(
            args.val_csv,
            trainer.processor,
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader, num_epochs=50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
