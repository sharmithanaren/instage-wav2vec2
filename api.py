import os
import time
import requests
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from fastapi import FastAPI, Form, File, UploadFile
from deepgram import DeepgramClient, SpeakOptions
from dotenv import load_dotenv
from pydantic import BaseModel


# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Load the processor and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("bookbot/wav2vec2-ljspeech-gruut")
model = Wav2Vec2ForCTC.from_pretrained("bookbot/wav2vec2-ljspeech-gruut").to(device)

print(device)

# ratio = model.config.inputs_to_logits_ratio
# sr = processor.feature_extractor.sampling_rate
# time_offset = ratio / sr

# Deepgram API details
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = 'https://api.deepgram.com/v1/speak?model=aura-helios-en'
headers = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

# Phoneme to viseme ID mapping based on phonemeObject
phonemeToVisemeIDMapping = {
    "Silence": 0, "[PAD]": 0, "[UNK]": 0, "|": 0,
    "æ": 1, "ə": 1, "ʌ": 1,
    "ɑ": 2,
    "ɔ": 3,
    "ɛ": 4, "ʊ": 4,
    "ɝ": 5, "ɚ": 5,
    "j": 6, "i": 6, "ɪ": 6,
    "w": 7, "u": 7,
    "o": 8, "oʊ": 8,
    "aʊ": 9,
    "ɔɪ": 10,
    "aɪ": 11,
    "h": 12,
    "ɹ": 13,
    "l": 14,
    "s": 15, "z": 15,
    "ʃ": 16, "tʃ": 16, "t͡ʃ": 16, "dʒ": 16, "d͡ʒ": 16, "ʒ": 16,
    "ð": 17,
    "f": 18, "v": 18,
    "d": 19, "t": 19, "n": 19, "θ": 19,
    "k": 20, "g": 20, "ɡ": 20, "ŋ": 20,
    "p": 21, "b": 21, "m": 21,
    "eɪ": 1,  # Same as "æ", "ə", "ʌ"
    "aɪ": 11,  # Same as "aɪ"
    "aʊ": 9,  # Same as "aʊ"
    "d͡ʒ": 16,  # Same as "dʒ", "ʒ"
    "oʊ": 8,  # Same as "o"
    "t͡ʃ": 16,  # Same as "tʃ", "ʃ"
    "ɔɪ": 10,  # Same as "ɔɪ"
    "ɝ": 5,  # Same as "ɝ"
    "ɡ": 20  # Same as "k", "g", "ŋ"
}

phonemeToVisemeMapping = {
    "Silence": "SIL",
    "æ": "A_E",
    "ə": "A_E",
    "ʌ": "A_E",
    "ɑ": "Ah",
    "ɔ": "Oh",
    "ɛ": "EE",
    "ʊ": "W_OO",
    "ɝ": "Er",
    "j": "EE",
    "i": "EE",
    "ɪ": "Ih",
    "w": "W_OO",
    "u": "W_OO",
    "o": "Oh",
    "aʊ": "W_OO",
    "ɔɪ": "Oh",
    "aɪ": "A_E",
    "h": "SIL",
    "ɹ": "R",
    "l": "T_L_D",
    "s": "S_Z",
    "z": "S_Z",
    "ʃ": "S_Z",
    "tʃ": "CH_J",
    "dʒ": "CH_J",
    "ʒ": "S_Z",
    "ð": "Th",
    "f": "F_V",
    "v": "F_V",
    "d": "T_L_D",
    "t": "T_L_D",
    "n": "T_L_D",
    "θ": "Th",
    "k": "K_G",
    "g": "K_G",
    "ŋ": "K_G",
    "p": "B_M_P",
    "b": "B_M_P",
    "m": "B_M_P",
    "[PAD]": "SIL",
    "[UNK]": "SIL",
    "d͡ʒ": "CH_J",
    "eɪ": "A_E",
    "oʊ": "Oh",
    "t͡ʃ": "CH_J",
    "|": "SIL",
    "ɚ": "Er",
    "ɡ": "K_G",
    "aɪ": "A_E",  # Already exists, no change needed
    "aʊ": "W_OO",  # Already exists, no change needed
    "oʊ": "Oh",  # Already exists, no change needed
    "d͡ʒ": "CH_J",  # Already exists, no change needed
    "t͡ʃ": "CH_J",  # Already exists, no change needed
    "ɔɪ": "Oh",  # Already exists, no change needed
}


# Function to map phonemes to viseme IDs and mappings with offsets
def map_phonemes_to_visemes_with_offsets(transcription):

    time_offset = 0.02

    visemes_with_offsets = []

    for item in transcription['char_offsets'][0]:
        phoneme = item['char']
        
        # Skip the current space if the previous phoneme was also a space
        if phoneme == " ":
            continue

        start_offset = item['start_offset']
        viseme_id = phonemeToVisemeIDMapping.get(phoneme, 0)  # Default to "Silence" ID if phoneme not found
        #viseme_mapping = phonemeToVisemeMapping.get(phoneme, "SIL")  # Default to "SIL" if phoneme not found
        
        
        visemes_with_offsets.append({
            'visemeId': int(viseme_id),  # Ensure it's a Python int
            #'visemeMapping': viseme_mapping,
            'audioOffset': int(round(start_offset * time_offset*1000, 4)) # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/tokenization_wav2vec2.py
        })
    
    return visemes_with_offsets

# Function to process a single audio file
def process_audio_file(audio_file_path):
    # Load and preprocess the audio file
    audio, rate = librosa.load(audio_file_path, sr=16000)

    # Tokenize the input audio
    inputs = processor(audio, return_tensors="pt", padding=True).to(device)

    # Start time measurement
    start = time.time()

    # Perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Decode the logits to text with character offsets
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, output_char_offsets=True)

    # End time measurement
    end = time.time()

    # Calculate time taken in milliseconds
    time_taken_ms = (end - start) * 1000

    # Map the phonemes to viseme IDs with offsets
    viseme_transcription_with_offsets = map_phonemes_to_visemes_with_offsets(transcription)
    
    return (audio_file_path, int(time_taken_ms), viseme_transcription_with_offsets)

# Function to synthesize audio using Deepgram
def synthesize_audio(text, output_file_path):
    payload = {"text": text}
    with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
        with open(output_file_path, "wb") as output_file:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    output_file.write(chunk)

class VisemeTranscription(BaseModel):
    visemeId: int
    audioOffset: int

class ProcessedAudioResponse(BaseModel):
    viseme_transcription_with_offsets: list[VisemeTranscription]

@app.post("/process_text", response_model=ProcessedAudioResponse)
async def process_text(input_text: str = Form(...)):
    # Create or truncate the output file
    temp_dir = 'temp_audio_files'
    os.makedirs(temp_dir, exist_ok=True)
    audio_file_path = os.path.join(temp_dir, "output.wav")

    # Synthesize audio for the entire text
    synthesize_audio(input_text, audio_file_path)
    
    # Process the synthesized audio file
    audio_file_path, time_taken_ms, viseme_transcription_with_offsets = process_audio_file(audio_file_path)

    os.remove(audio_file_path)
    
    return {
        'audio_file': os.path.basename(audio_file_path),
        'time_taken_ms': time_taken_ms,
        'viseme_transcription_with_offsets': viseme_transcription_with_offsets
    }
@app.post("/process_audio")
async def process_audio(audio: UploadFile = File(...)):
    audio_file = audio
    temp_dir = 'temp_audio_files'
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, audio.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await audio.read())

    audio_file_path, time_taken_ms, viseme_transcription_with_offsets = process_audio_file(file_path)

    os.remove(file_path)
    
    return viseme_transcription_with_offsets

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
