
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
