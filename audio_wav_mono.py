from pydub import AudioSegment
import os

def convert_to_wav_mono(input_audio_path, output_wav_path):
    """
    Convert any audio file to .wav format and ensure it is mono.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_wav_path (str): Path to save the output .wav file.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_audio_path)

        # Convert to mono if it's not already
        if audio.channels > 1:
            print(f"Converting {os.path.basename(input_audio_path)} to mono...")
            audio = audio.set_channels(1)  # Convert to mono

        # Export as .wav file
        audio.export(output_wav_path, format="wav")
        print(f"Converted {os.path.basename(input_audio_path)} to mono .wav format: {output_wav_path}")

    except Exception as e:
        print(f"Error processing {os.path.basename(input_audio_path)}: {e}")

def process_audio_folder(input_folder, output_folder):
    """
    Process all audio files in the input folder and save converted files to the output folder.

    Args:
        input_folder (str): Path to the folder containing input audio files.
        output_folder (str): Path to the folder to save converted .wav files.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Get a list of all audio files in the input folder
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a'))]

    if not audio_files:
        print(f"No audio files found in the input folder: {input_folder}")
        return

    print(f"Found {len(audio_files)} audio files to process...")

    # Process each audio file
    for audio_file in audio_files:
        input_audio_path = os.path.join(input_folder, audio_file)
        output_wav_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + ".wav")
        convert_to_wav_mono(input_audio_path, output_wav_path)

if __name__ == "__main__":
    # Input and output folder paths
    input_folder = input("Enter the path to the input folder containing audio files: ").strip()
    output_folder = input("Enter the path to the output folder for converted .wav files: ").strip()

    # Process all audio files in the folder
    process_audio_folder(input_folder, output_folder)