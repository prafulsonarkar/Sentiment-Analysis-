import os
import random
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup, pitch_shift
from scipy.io.wavfile import read, write

def change_speed(audio, speed_factor):
    """
    Change the speed of the audio without changing the pitch.
    
    Args:
        audio (AudioSegment): Input audio.
        speed_factor (float): Speed factor (e.g., 1.1 for 10% faster, 0.9 for 10% slower).
    
    Returns:
        AudioSegment: Audio with modified speed.
    """
    return speedup(audio, playback_speed=speed_factor)

def shift_pitch(audio, pitch_shift_steps):
    """
    Shift the pitch of the audio.
    
    Args:
        audio (AudioSegment): Input audio.
        pitch_shift_steps (float): Number of steps to shift the pitch (e.g., 2 for higher pitch, -2 for lower pitch).
    
    Returns:
        AudioSegment: Audio with shifted pitch.
    """
    return pitch_shift(audio, pitch_shift_steps)

def adjust_volume(audio, volume_change_db):
    """
    Adjust the volume of the audio.
    
    Args:
        audio (AudioSegment): Input audio.
        volume_change_db (float): Volume change in decibels (e.g., 10 for louder, -10 for quieter).
    
    Returns:
        AudioSegment: Audio with adjusted volume.
    """
    return audio + volume_change_db

def add_background_noise(audio, noise_file, noise_level_db=-20):
    """
    Add background noise to the audio.
    
    Args:
        audio (AudioSegment): Input audio.
        noise_file (str): Path to the background noise file.
        noise_level_db (float): Noise level in decibels relative to the audio.
    
    Returns:
        AudioSegment: Audio with added background noise.
    """
    noise = AudioSegment.from_file(noise_file)
    noise = noise - abs(noise_level_db)  # Reduce noise volume
    combined = audio.overlay(noise)  # Mix audio and noise
    return combined

def augment_audio(input_file, output_file, noise_file=None):
    """
    Apply random augmentations to the audio file.
    
    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the augmented audio file.
        noise_file (str, optional): Path to the background noise file.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Randomly apply augmentations
    if random.random() < 0.5:  # 50% chance to change speed
        speed_factor = random.uniform(0.9, 1.1)  # Speed change between 90% and 110%
        audio = change_speed(audio, speed_factor)

    if random.random() < 0.5:  # 50% chance to shift pitch
        pitch_shift_steps = random.uniform(-2, 2)  # Pitch shift between -2 and +2 steps
        audio = shift_pitch(audio, pitch_shift_steps)

    if random.random() < 0.5:  # 50% chance to adjust volume
        volume_change_db = random.uniform(-10, 10)  # Volume change between -10 dB and +10 dB
        audio = adjust_volume(audio, volume_change_db)

    if noise_file and random.random() < 0.5:  # 50% chance to add background noise
        audio = add_background_noise(audio, noise_file)

    # Export the augmented audio
    audio.export(output_file, format="wav")
    print(f"Augmented audio saved to: {output_file}")

def augment_audio_folder(input_folder, output_folder, noise_file=None):
    """
    Augment all audio files in the input folder and save them to the output folder.
    
    Args:
        input_folder (str): Path to the folder containing input audio files.
        output_folder (str): Path to the folder to save augmented audio files.
        noise_file (str, optional): Path to the background noise file.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Get a list of all audio files in the input folder
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    if not audio_files:
        print(f"No audio files found in the input folder: {input_folder}")
        return

    print(f"Found {len(audio_files)} audio files to augment...")

    # Process each audio file
    for audio_file in audio_files:
        input_file = os.path.join(input_folder, audio_file)
        output_file = os.path.join(output_folder, f"augmented_{audio_file}")
        augment_audio(input_file, output_file, noise_file)

if __name__ == "__main__":
    # Input and output folder paths
    input_folder = input("Enter the path to the input folder containing audio files: ").strip()
    output_folder = input("Enter the path to the output folder for augmented audio files: ").strip()
    noise_file = input("Enter the path to the background noise file (optional, press Enter to skip): ").strip() or None

    # Augment all audio files in the folder
    augment_audio_folder(input_folder, output_folder, noise_file)