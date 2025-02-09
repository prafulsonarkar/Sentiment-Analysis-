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
import wandb
from tqdm import tqdm
import jiwer
from typing import Dict, Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """Dataset for mixed language audio data"""
    def __init__(self, csv_path: str, processor: WhisperProcessor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        logger.info(f"Loaded {len(self.df)} examples from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Load and process audio
            waveform, sample_rate = torchaudio.load(row['audio_path'])
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Process audio features
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Process transcription
            labels = self.processor(
                text=row['transcription'],
                return_tensors="pt"
            ).input_ids
            
            return {
                'input_features': inputs.input_features.squeeze(),
                'labels': labels.squeeze(),
                'transcription': row['transcription']
            }
        except Exception as e:
            logger.error(f"Error processing {row['audio_path']}: {str(e)}")
            raise

class CompactStudentModel(nn.Module):
    """Compact student model for mixed language ASR"""
    def __init__(self, vocab_size: int, hidden_dim: int = 512):
        super().__init__()
        
        # Audio feature encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(80, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=8
            )
        )
        
        # Language-specific attention
        self.hindi_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)
        self.mixed_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=8
        )
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, audio_features: torch.Tensor, decoder_input: torch.Tensor, 
               decoder_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode audio
        encoded = self.encoder(audio_features.transpose(1, 2)).transpose(1, 2)
        
        # Apply language-specific attention
        hindi_features, _ = self.hindi_attention(encoded, encoded, encoded)
        mixed_features, _ = self.mixed_attention(encoded, encoded, encoded)
        
        # Combine features
        memory = hindi_features + mixed_features
        
        # Decode
        tgt_embeddings = self.embedding(decoder_input)
        decoded = self.decoder(tgt_embeddings, memory, tgt_mask=decoder_mask)
        output = self.output_layer(decoded)
        
        return output
    
    def generate(self, features: torch.Tensor, max_length: int = 448) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        
        # Start token
        decoder_input = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        
        # Encode input
        encoded = self.encoder(features.transpose(1, 2)).transpose(1, 2)
        hindi_features, _ = self.hindi_attention(encoded, encoded, encoded)
        mixed_features, _ = self.mixed_attention(encoded, encoded, encoded)
        memory = hindi_features + mixed_features
        
        # Generate tokens
        for _ in range(max_length - 1):
            mask = self._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            tgt_embeddings = self.embedding(decoder_input)
            decoded = self.decoder(tgt_embeddings, memory, tgt_mask=mask)
            output = self.output_layer(decoded)
            
            next_token = output[:, -1:].argmax(dim=-1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            if (next_token == 50257).all():  # End token
                break
                
        return decoder_input
    
    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask

class MultilingualTrainer:
    """Trainer for multilingual knowledge distillation"""
    def __init__(
        self,
        teacher_model: str = "vasista/whisper-hindi-large-v2",
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        temperature: float = 2.5,
        distillation_alpha: float = 0.6,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.distillation_alpha = distillation_alpha
        
        # Initialize models
        self.processor = WhisperProcessor.from_pretrained(teacher_model)
        self.teacher = WhisperForConditionalGeneration.from_pretrained(teacher_model)
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
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_dir: str,
        checkpoint_path: Optional[str] = None
    ):
        # Resume from checkpoint if provided
        start_epoch = 0
        best_wer = float('inf')
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.student.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_wer = checkpoint['best_wer']
            logger.info(f"Resumed from epoch {start_epoch}")
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(start_epoch, epochs):
            # Training
            self.student.train()
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                loss = self._training_step(batch)
                train_losses.append(loss)
            
            # Validation
            val_metrics = self._validate(val_loader)
            val_wer = val_metrics['wer']
            
            # Save checkpoint
            if val_wer < best_wer:
                best_wer = val_wer
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_wer': best_wer,
                }, f"{save_dir}/best_model.pth")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_wer': best_wer,
                }, f"{save_dir}/checkpoint_epoch_{epoch}.pth")
            
            logger.info(f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_wer={val_wer:.4f}")
    
    def _training_step(self, batch):
        audio_features = batch['input_features'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = self.teacher(audio_features, labels=labels)
        
        # Student forward pass
        student_outputs = self.student(
            audio_features,
            labels[:, :-1],
            self.student._generate_square_subsequent_mask(labels.size(1) - 1).to(self.device)
        )
        
        # Calculate losses
        distill_loss = self.kl_loss(
            torch.log_softmax(student_outputs / self.temperature, dim=-1),
            torch.softmax(teacher_outputs.logits[:, :-1] / self.temperature, dim=-1)
        )
        
        ce_loss = self.ce_loss(
            student_outputs.view(-1, student_outputs.size(-1)),
            labels[:, 1:].contiguous().view(-1)
        )
        
        # Combined loss
        loss = (self.distillation_alpha * distill_loss + 
                (1 - self.distillation_alpha) * ce_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _validate(self, val_loader):
        self.student.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in val_loader:
                audio_features = batch['input_features'].to(self.device)
                generated_ids = self.student.generate(audio_features)
                
                # Decode predictions
                pred_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(pred_texts)
                references.extend(batch['transcription'])
        
        wer = jiwer.wer(references, predictions)
        return {'wer': wer, 'predictions': predictions, 'references': references}
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file"""
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        self.student.eval()
        with torch.no_grad():
            generated_ids = self.student.generate(
                inputs.input_features.to(self.device)
            )
        
        transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return transcription

def main():
    # Configuration
    config = {
        'train_csv': 'path/to/your/train.csv',  # Update with your CSV path
        'val_csv': 'path/to/your/val.csv',      # Update with your CSV path
        'save_dir': 'checkpoints',
        'epochs': 50,
        'batch_size': 16,
        'hidden_dim': 512,
        'learning_rate': 1e-4,
        'temperature': 2.5,
        'distillation_alpha': 0.6,
        'checkpoint_path': None  # Set to checkpoint path if resuming training
    }
    
    # Initialize trainer
    trainer = MultilingualTrainer(
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
