import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
import csv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_and_save_model(model_id, save_path):
    """
    Download and save the model locally
    """
    logging.info(f"Downloading model {model_id} to {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Download and save model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        local_files_only=False,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.save_pretrained(save_path)
    
    # Download and save processor
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
    processor.save_pretrained(save_path)
    
    logging.info("Model and processor saved successfully!")
    return save_path

def setup_transcription_pipeline(model_path, language="hi"):
    """
    Set up the transcription pipeline
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    transcribe = pipeline(
        task="automatic-speech-recognition",
        model=model_path,
        chunk_length_s=30,
        device=device
    )
    
    transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(
        language=language,
        task="transcribe"
    )
    
    return transcribe

def process_audio_file(transcribe, audio_path):
    """
    Process a single audio file and return its transcription
    """
    try:
        result = transcribe(str(audio_path))
        return result["text"]
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")
        return None

def save_transcriptions_csv(transcriptions, output_path):
    """
    Save transcriptions to a CSV file with only file path and transcription
    """
    output_file = output_path / "transcriptions.csv"
    
    # Define CSV headers
    headers = ['File Path', 'Transcription']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for file_path, transcription in transcriptions.items():
            writer.writerow([file_path, transcription])
    
    logging.info(f"Transcriptions saved to {output_file}")
    return output_file

def main():
    # Configuration
    MODEL_ID = "vasista22/whisper-hindi-large-v2"
    MODEL_PATH = "./local_whisper_model"
    AUDIO_FOLDER = Path("./audio_files")  # Replace with your audio folder path
    OUTPUT_PATH = Path("./transcriptions")
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac'}
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Download model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        download_and_save_model(MODEL_ID, MODEL_PATH)
    
    # Setup transcription pipeline
    transcribe = setup_transcription_pipeline(MODEL_PATH)
    
    # Process all audio files
    transcriptions = {}
    total_files = sum(1 for f in AUDIO_FOLDER.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS)
    
    logging.info(f"Found {total_files} audio files to process")
    
    # Process each file
    for i, audio_file in enumerate(AUDIO_FOLDER.iterdir(), 1):
        if audio_file.suffix.lower() in SUPPORTED_FORMATS:
            logging.info(f"Processing file {i}/{total_files}: {audio_file.name}")
            
            transcription = process_audio_file(transcribe, audio_file)
            
            if transcription:
                transcriptions[str(audio_file)] = transcription
    
    # Save results
    if transcriptions:
        output_file = save_transcriptions_csv(transcriptions, OUTPUT_PATH)
        logging.info(f"Successfully processed {len(transcriptions)} files")
    else:
        logging.warning("No transcriptions were generated")

if __name__ == "__main__":
    main()
