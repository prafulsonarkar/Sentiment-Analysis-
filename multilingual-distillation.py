# [Previous imports remain the same...]

class AudioDataset(Dataset):
    def __init__(self, csv_path: str, processor: WhisperProcessor, max_audio_length: int = 30):
        """
        Dataset for loading audio and transcriptions from CSV
        
        Args:
            csv_path: Path to CSV file with audio_path and transcription columns
            processor: WhisperProcessor instance
            max_audio_length: Maximum audio length in seconds
        """
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_audio_length = max_audio_length
        
        logger.info(f"Loaded {len(self.df)} examples from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # Load and process audio
            waveform, sample_rate = torchaudio.load(row['audio_path'])
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Truncate if too long
            max_length = self.max_audio_length * 16000  # Convert seconds to samples
            if waveform.shape[1] > max_length:
                waveform = waveform[:, :max_length]
            
            # Process audio
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Process text
            labels = self.processor(
                text=row['transcription'],
                return_tensors="pt"
            ).input_ids
            
            return {
                'input_features': inputs.input_features.squeeze(),
                'labels': labels.squeeze()
            }
            
        except Exception as e:
            logger.error(f"Error processing {row['audio_path']}: {str(e)}")
            raise

def create_dataloaders(
    train_csv: str,
    val_csv: str,
    processor: WhisperProcessor,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = None
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from CSV files
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        processor: WhisperProcessor instance
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        device: Device to use ('cuda' or 'cpu')
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = AudioDataset(train_csv, processor)
    val_dataset = AudioDataset(val_csv, processor)
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    
    return train_loader, val_loader

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize trainer
    trainer = MultilingualDistillationTrainer(
        teacher_model_name="vasista/whisper-hindi-large-v2",
        hidden_dim=384,
        learning_rate=2e-4,
        max_epochs=50,
        batch_size=16,
        gradient_accumulation_steps=2,
        temperature=2.0,
        distillation_alpha=0.5,
        device=device
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_csv="path/to/train.csv",
        val_csv="path/to/val.csv",
        processor=trainer.processor,
        batch_size=16,
        num_workers=4,
        device=device
    )
    
    # Train the model
    trainer.train(train_loader, val_loader, max_epochs=50)
    
    # Example transcription
    transcription = trainer.transcribe("path/to/test_audio.wav")
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()
