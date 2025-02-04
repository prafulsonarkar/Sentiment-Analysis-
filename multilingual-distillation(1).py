import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompactWhisperEncoder(nn.Module):
    """Compact student encoder optimized for Hindi and related languages"""
    def __init__(
        self, 
        input_dim: int = 80,
        hidden_dim: int = 384,  # Reduced from teacher's dimension
        num_layers: int = 6,
        num_heads: int = 6
    ):
        super().__init__()
        
        # Initial feature processing
        self.feature_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Specialized layers for different language aspects
        self.hindi_specific = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        self.mixed_language = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        
        # Main transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Language-mixing layer
        self.language_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project features
        x = self.feature_projection(x.transpose(1, 2)).transpose(1, 2)
        
        # Apply Hindi-specific attention
        hindi_features, _ = self.hindi_specific(x, x, x)
        
        # Apply mixed-language attention
        mixed_features, _ = self.mixed_language(x, x, x)
        
        # Combine features
        combined = hindi_features + mixed_features
        
        # Main encoding
        encoded = self.encoder(combined)
        
        # Final language mixing
        output = self.language_mixer(encoded)
        
        return output

class CompactWhisperDecoder(nn.Module):
    """Compact student decoder optimized for Hindi output"""
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6
    ):
        super().__init__()
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embed target tokens
        tgt_embeddings = self.embed(tgt)
        
        # Decode
        decoded = self.decoder(tgt_embeddings, memory, tgt_mask=tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoded)
        
        return output

class CompactStudent(nn.Module):
    """Complete student model"""
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        hidden_dim: int = 384,
        encoder_layers: int = 6,
        decoder_layers: int = 6
    ):
        super().__init__()
        
        self.encoder = CompactWhisperEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers
        )
        
        self.decoder = CompactWhisperDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers
        )
        
    def forward(
        self,
        audio_features: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoder_output = self.encoder(audio_features)
        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask)
        return decoder_output

class MultilingualDistillationTrainer:
    def __init__(
        self,
        teacher_model_name: str = "vasista/whisper-hindi-large-v2",
        hidden_dim: int = 384,
        learning_rate: float = 2e-4,
        max_epochs: int = 50,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 2,
        temperature: float = 2.0,
        distillation_alpha: float = 0.5,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.distillation_alpha = distillation_alpha
        
        # Load teacher model and processor
        self.processor = WhisperProcessor.from_pretrained(teacher_model_name)
        self.teacher = WhisperForConditionalGeneration.from_pretrained(teacher_model_name)
        self.teacher.eval()
        self.teacher.to(self.device)
        
        # Create student model
        self.student = CompactStudent(
            vocab_size=self.processor.tokenizer.vocab_size,
            hidden_dim=hidden_dim
        )
        self.student.to(self.device)
        
        # Training components
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=max_epochs,
            steps_per_epoch=1000 // batch_size // gradient_accumulation_steps,
            pct_start=0.1
        )
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.student.train()
        
        # Move batch to device
        audio_features = batch['input_features'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = self.teacher(
                audio_features,
                labels=labels,
                output_hidden_states=True
            )
            
        # Student forward pass
        student_outputs = self.student(
            audio_features,
            labels[:, :-1],  # Remove last token for teacher forcing
            self.generate_square_subsequent_mask(labels.size(1) - 1).to(self.device)
        )
        
        # Calculate losses
        # 1. Distillation loss
        distill_loss = self.kl_loss(
            torch.log_softmax(student_outputs / self.temperature, dim=-1),
            torch.softmax(teacher_outputs.logits[:, :-1] / self.temperature, dim=-1)
        )
        
        # 2. Cross entropy loss
        ce_loss = self.ce_loss(
            student_outputs.view(-1, student_outputs.size(-1)),
            labels[:, 1:].contiguous().view(-1)  # Shift right for teacher forcing
        )
        
        # Combined loss
        loss = (
            self.distillation_alpha * distill_loss +
            (1 - self.distillation_alpha) * ce_loss
        )
        
        return {
            'loss': loss,
            'distill_loss': distill_loss.item(),
            'ce_loss': ce_loss.item()
        }
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate mask for decoder self-attention"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        max_epochs: int,
        gradient_accumulation_steps: int = 2
    ):
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training loop
            self.student.train()
            train_losses = []
            
            for i, batch in enumerate(train_dataloader):
                losses = self.train_step(batch)
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                train_losses.append(losses['loss'].item())
            
            # Validation
            val_loss = self.evaluate(val_dataloader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.student.state_dict(), 'best_student_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            logger.info(f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_loss={val_loss:.4f}")
    
    def evaluate(self, dataloader: DataLoader) -> float:
        self.student.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                losses = self.train_step(batch)
                val_losses.append(losses['loss'].item())
        
        return np.mean(val_losses)
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using the student model"""
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Process audio
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Generate transcription
        self.student.eval()
        with torch.no_grad():
            generated_ids = self.student.generate(
                inputs.input_features.to(self.device),
                max_length=448
            )
            
        transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return transcription

def main():
    # Initialize trainer with optimal hyperparameters for your dataset
    trainer = MultilingualDistillationTrainer(
        teacher_model_name="vasista/whisper-hindi-large-v2",
        hidden_dim=384,            # Compact but effective size
        learning_rate=2e-4,        # Slightly higher for faster adaptation
        max_epochs=50,
        batch_size=16,             # Adjust based on GPU memory
        gradient_accumulation_steps=2,
        temperature=2.0,           # Good for knowledge transfer
        distillation_alpha=0.5     # Equal weight to distillation and CE loss
    )
    
    # Load your data and create dataloaders
    # train_dataloader = ...
    # val_dataloader = ...
    
    # Train the model
    trainer.train(train_dataloader, val_dataloader, max_epochs=50)
    
    # Example transcription
    transcription = trainer.transcribe("path/to/audio.wav")
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()
