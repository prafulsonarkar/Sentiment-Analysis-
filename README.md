
https://docs.google.com/document/d/1JcucSeNe5BkBydsPW_2nhq17LK8AgIaGjDjeNcdrBoU/edit?usp=drivesdk
## sentiment Analysis 
https://colab.research.google.com/drive/1hT4f2fcn37CSNx7VBBlc4bzpzk-plC8g#scrollTo=Yh132nFg9J_K
## Big Data Notes
https://docs.google.com/document/d/1CqYrLEIqRqMEJ_JGsHkyMeIiG4CjURaO/edit
## MHA Project Pipe-line Image
![MHA Project Flow Chart](https://github.com/user-attachments/assets/c34be7d9-5672-462e-955c-de6ae6fc3c2c)
## AMMIG
https://colab.research.google.com/drive/19DDqRjgXKY6yXlvsfCISz87ppj3Rt4iI#scrollTo=nmbCYw26s8x5

## Laion400M Dataset Link
https://drive.google.com/file/d/1BCWvfBSXDQNwDJAoEE42mXXBiubhXYH8/view?usp=drive_link

https://drive.google.com/file/d/1BCWvfBSXDQNwDJAoEE42mXXBiubhXYH8/view?usp=drivesdk
https://drive.google.com/file/d/1BCWvfBSXDQNwDJAoEE42mXXBiubhXYH8/view?usp=drivesdk


## Assembly ai
https://archive.ph/wMSX0

## code pretrain model

https://docs.google.com/document/d/1d5rpajRKFnLFUbrWSF0I7qwk8g67U9wuIIIxzC4TMik/edit?usp=drivesdk
https://docs.google.com/document/d/1d5rpajRKFnLFUbrWSF0I7qwk8g67U9wuIIIxzC4TMik/edit?usp=drivesdk
## code for whisper large v2 fine tuned model
import torch
import torchaudio
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import List, Dict, Union
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime

class AudioTranscriber:
    def __init__(
        self, 
        model_path: str,
        device: str = None,
        batch_size: int = 8,
        sample_rate: int = 16000
    ):
        """
        Initialize the transcriber with a fine-tuned model
        
        Args:
            model_path: Path to fine-tuned model
            device: 'cuda' or 'cpu'
            batch_size: Batch size for processing multiple files
            sample_rate: Target sample rate for audio
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        
        # Load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Model loaded on {self.device}")
        
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess a single audio file
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.sample_rate
                )
                waveform = resampler(waveform)
            
            return waveform.squeeze().numpy()
            
        except Exception as e:
            self.logger.error(f"Error processing {audio_path}: {str(e)}")
            raise
    
    def transcribe_audio(
        self, 
        audio_path: str,
        return_timestamps: bool = False
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            return_timestamps: Whether to return word-level timestamps
        
        Returns:
            Dictionary containing transcription and optional timestamps
        """
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Prepare inputs
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                if return_timestamps:
                    outputs = self.model.generate(
                        input_features,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_length=448,
                        output_attentions=True
                    )
                else:
                    outputs = self.model.generate(input_features)
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                outputs if not return_timestamps else outputs.sequences,
                skip_special_tokens=True
            )[0]
            
            result = {"transcription": transcription}
            
            # Add timestamps if requested
            if return_timestamps:
                # Process attention weights for word-level timestamps
                timestamps = self._extract_timestamps(outputs, audio.shape[0])
                result["timestamps"] = timestamps
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error transcribing {audio_path}: {str(e)}")
            raise
    
    def batch_transcribe(
        self, 
        audio_folder: str,
        output_path: str = None,
        file_pattern: str = "*.wav",
        return_timestamps: bool = False
    ) -> pd.DataFrame:
        """
        Transcribe all audio files in a folder
        
        Args:
            audio_folder: Path to folder containing audio files
            output_path: Path to save results (optional)
            file_pattern: Pattern to match audio files
            return_timestamps: Whether to include timestamps
        
        Returns:
            DataFrame with transcription results
        """
        # Get list of audio files
        audio_files = list(Path(audio_folder).glob(file_pattern))
        self.logger.info(f"Found {len(audio_files)} audio files to process")
        
        results = []
        
        # Process files in batches
        for i in tqdm(range(0, len(audio_files), self.batch_size)):
            batch_files = audio_files[i:i + self.batch_size]
            
            for audio_file in batch_files:
                try:
                    result = self.transcribe_audio(
                        str(audio_file),
                        return_timestamps=return_timestamps
                    )
                    
                    result['audio_file'] = audio_file.name
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {audio_file}: {str(e)}")
                    results.append({
                        'audio_file': audio_file.name,
                        'transcription': 'ERROR',
                        'error': str(e)
                    })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results if output path provided
        if output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save as CSV
            csv_path = f"{output_path}/transcriptions_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            # Save detailed results as JSON if timestamps included
            if return_timestamps:
                json_path = f"{output_path}/transcriptions_detailed_{timestamp}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Results saved to {csv_path}")
        
        return df
    
    def _extract_timestamps(self, outputs, audio_length):
        """Extract word-level timestamps from model outputs"""
        # This is a simplified version. Actual implementation would depend
        # on your specific needs and model output format
        attention_weights = outputs.attentions[-1]
        
        # Convert attention weights to timestamps
        audio_duration = audio_length / self.sample_rate
        timestamps = []
        
        # Process attention weights to get word timings
        # This is a placeholder - implement based on your needs
        return timestamps

def main():
    """Example usage"""
    # Initialize transcriber
    transcriber = AudioTranscriber(
        model_path="path/to/your/finetuned/model",  # Replace with your model path
        batch_size=8
    )
    
    # Single file transcription
    result = transcriber.transcribe_audio(
        "path/to/audio.wav",
        return_timestamps=True
    )
    print(f"Transcription: {result['transcription']}")
    
    # Batch transcription
    results_df = transcriber.batch_transcribe(
        audio_folder="path/to/audio/folder",
        output_path="path/to/output",
        file_pattern="*.wav",
        return_timestamps=True
    )
    print("\nBatch processing results:")
    print(results_df.head())

if __name__ == "__main__":
    main()
## Information Distilled
import torch
import torch.nn as nn
from torch.nn import functional as F
import editdistance
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import re

class TransliterationLayer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Mapping layer for non-Hindi to Hindi characters
        self.char_mapping = nn.Linear(vocab_size, vocab_size)
        
    def forward(self, x):
        # Apply character mapping
        return self.char_mapping(x)

class HindiTranscriptionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = IndicNormalizerFactory().get_normalizer('hi')
        
    def forward(self, pred, target, language_tags):
        """
        Custom loss for Hindi transcription with transliteration awareness
        """
        base_loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
        
        # Additional penalty for non-Hindi characters in output
        hindi_char_mask = (target >= self.hindi_char_start) & (target < self.hindi_char_end)
        transliteration_loss = F.cross_entropy(
            pred.view(-1, pred.size(-1))[hindi_char_mask],
            target.view(-1)[hindi_char_mask]
        ) if torch.any(hindi_char_mask) else 0
        
        return base_loss + 0.5 * transliteration_loss

class MetricsCalculator:
    def __init__(self, processor):
        self.processor = processor
        self.normalizer = IndicNormalizerFactory().get_normalizer('hi')
        
    def normalize_text(self, text):
        """Normalize Hindi text removing extra spaces and unwanted characters"""
        text = self.normalizer.normalize(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def transliterate_to_hindi(self, text, source_lang):
        """Transliterate text to Hindi if needed"""
        if source_lang in ['pa', 'ur']:
            try:
                if source_lang == 'pa':
                    text = transliterate(text, sanscript.GURMUKHI, sanscript.DEVANAGARI)
                elif source_lang == 'ur':
                    text = transliterate(text, sanscript.URDU, sanscript.DEVANAGARI)
            except:
                pass  # Keep original text if transliteration fails
        return text
    
    def calculate_wer(self, hypothesis, reference, language_tags):
        """
        Calculate Word Error Rate with language-aware processing
        """
        # Normalize and transliterate both hypothesis and reference
        hyp_words = []
        ref_words = []
        
        for h, r, langs in zip(hypothesis, reference, language_tags):
            # Decode if needed
            if isinstance(h, torch.Tensor):
                h = self.processor.batch_decode(h, skip_special_tokens=True)[0]
            if isinstance(r, torch.Tensor):
                r = self.processor.batch_decode(r, skip_special_tokens=True)[0]
            
            # Process each language segment
            for lang in langs:
                h = self.transliterate_to_hindi(h, lang)
                r = self.transliterate_to_hindi(r, lang)
            
            # Normalize
            h = self.normalize_text(h)
            r = self.normalize_text(r)
            
            hyp_words.extend(h.split())
            ref_words.extend(r.split())
        
        # Calculate WER
        distance = editdistance.eval(hyp_words, ref_words)
        return (distance / len(ref_words)) * 100

def train_with_metrics(
    teacher_model,
    student_model,
    train_loader,
    valid_loader,
    num_epochs=20,
    temp=2.0,
    alpha=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Initialize losses
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    feat_criterion = nn.MSELoss()
    hindi_criterion = HindiTranscriptionLoss()
    metrics_calculator = MetricsCalculator(processor)
    
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_valid_wer = float('inf')
    
    for epoch in range(num_epochs):
        student_model.train()
        train_wer = 0
        train_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            language_tags = batch['language_tags']
            
            # Teacher predictions
            with torch.no_grad():
                teacher_output = teacher_model(
                    input_features,
                    labels=labels,
                    output_hidden_states=True
                )
                teacher_logits = teacher_output.logits
                teacher_features = teacher_output.hidden_states[-1]
            
            # Student forward pass
            student_output = student_model(input_features, language_tags)
            
            # Calculate losses
            kl_loss = kl_criterion(
                torch.log_softmax(student_output / temp, dim=-1),
                torch.softmax(teacher_logits / temp, dim=-1)
            )
            
            feat_loss = feat_criterion(student_output, teacher_features)
            hindi_loss = hindi_criterion(student_output, labels, language_tags)
            
            # Combined loss
            total_loss = (
                0.4 * kl_loss +
                0.3 * feat_loss +
                0.3 * hindi_loss
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            
            # Calculate WER for this batch
            with torch.no_grad():
                predictions = torch.argmax(student_output, dim=-1)
                batch_wer = metrics_calculator.calculate_wer(
                    predictions, labels, language_tags
                )
                train_wer += batch_wer
                train_loss += total_loss.item()
                num_batches += 1
        
        # Validation phase
        student_model.eval()
        valid_wer = 0
        valid_loss = 0
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                input_features = batch['input_features'].to(device)
                labels = batch['labels'].to(device)
                language_tags = batch['language_tags']
                
                student_output = student_model(input_features, language_tags)
                predictions = torch.argmax(student_output, dim=-1)
                
                # Calculate validation WER
                batch_wer = metrics_calculator.calculate_wer(
                    predictions, labels, language_tags
                )
                valid_wer += batch_wer
                num_valid_batches += 1
        
        # Calculate epoch metrics
        epoch_train_wer = train_wer / num_batches
        epoch_valid_wer = valid_wer / num_valid_batches
        epoch_train_loss = train_loss / num_batches
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Train WER: {epoch_train_wer:.2f}%")
        print(f"Validation WER: {epoch_valid_wer:.2f}%")
        
        # Save best model
        if epoch_valid_wer < best_valid_wer:
            best_valid_wer = epoch_valid_wer
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_wer': best_valid_wer,
            }, 'best_hindi_asr_model.pth')
        
        scheduler.step()

def evaluate_model(model, test_loader, metrics_calculator):
    """
    Evaluate model with detailed metrics
    """
    model.eval()
    total_wer = 0
    language_specific_wer = {'hi': [], 'pa': [], 'ur': []}
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            language_tags = batch['language_tags']
            
            outputs = model(input_features, language_tags)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Calculate overall WER
            batch_wer = metrics_calculator.calculate_wer(
                predictions, labels, language_tags
            )
            total_wer += batch_wer
            
            # Calculate language-specific WER
            for i, tags in enumerate(language_tags):
                for lang in tags:
                    if lang in language_specific_wer:
                        lang_wer = metrics_calculator.calculate_wer(
                            predictions[i:i+1], 
                            labels[i:i+1], 
                            [tags]
                        )
                        language_specific_wer[lang].append(lang_wer)
            
            num_batches += 1
    
    # Calculate average metrics
    avg_wer = total_wer / num_batches
    lang_wer_results = {
        lang: sum(wers) / len(wers) if wers else 0 
        for lang, wers in language_specific_wer.items()
    }
    
    return {
        'overall_wer': avg_wer,
        'language_wer': lang_wer_results
    }

if __name__ == "__main__":
    # Initialize models and data loaders as before
    # Add evaluation code
    metrics_calculator = MetricsCalculator(processor)
    
    train_with_metrics(
        teacher_model,
        student_model,
        train_loader,
        valid_loader
    )
    
    # Final evaluation
    test_metrics = evaluate_model(student_model, test_loader, metrics_calculator)
    print("\nFinal Test Results:")
    print(f"Overall WER: {test_metrics['overall_wer']:.2f}%")
    print("\nLanguage-specific WER:")
    for lang, wer in test_metrics['language_wer'].items():
        print(f"{lang}: {wer:.2f}%")
