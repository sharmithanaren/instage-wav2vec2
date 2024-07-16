import streamlit as st
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Audio to Viseme Transcription",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 40px;
    }
    .upload {
        display: flex;
        justify-content: center;
        margin-bottom: 40px;
    }
    .audio {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .time-taken {
        font-size: 1.2em;
        font-weight: bold;
        color: #FF5722;
        text-align: center;
        margin-top: 20px;
    }
    .transcription {
        font-size: 1.2em;
        color: #2196F3;
        margin-top: 20px;
    }
    .json-output {
        font-size: 1em;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description of the app
st.markdown('<div class="title">Audio to Viseme Transcription</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="description">
        This application allows you to upload or record an audio file, which will be processed to generate a viseme transcription with offsets.
    </div>
    """,
    unsafe_allow_html=True
)

# Load the processor and model directly for more control
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("bookbot/wav2vec2-ljspeech-gruut")
    model = Wav2Vec2ForCTC.from_pretrained("bookbot/wav2vec2-ljspeech-gruut")
    return processor, model

processor, model = load_model()

# Function to map phonemes to visemes with offsets
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
    "[PAD]": "SIL",  # Assuming pad token is treated as silence
    "[UNK]": "SIL",  # Assuming unknown token is treated as silence
    "d͡ʒ": "CH_J",
    "eɪ": "A_E",
    "oʊ": "Oh",
    "t͡ʃ": "CH_J",
    "|": "SIL",  # Assuming this is a separator or silence
    "ɚ": "Er",
    "ɡ": "K_G"
}

def map_phonemes_to_visemes_with_offsets(transcription):
    visemes_with_offsets = []
    for item in transcription['char_offsets'][0]:
        phoneme = item['char']
        if phoneme == " ":  # Skip if the phoneme is a space
            continue
        start_offset = item['start_offset']
        end_offset = item['end_offset']
        viseme = phonemeToVisemeMapping.get(phoneme, "Not Found")  # Default to "Not Found" if phoneme not found
        visemes_with_offsets.append({
            'viseme': viseme,
            'start_offset': start_offset,
            'end_offset': end_offset
        })
    return visemes_with_offsets

def process_audio_chunks(audio, rate, chunk_duration=5):
    # Calculate the number of samples per chunk
    chunk_samples = int(chunk_duration * rate)
    total_samples = len(audio)
    visemes_with_offsets = []
    
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        audio_chunk = audio[start:end]

        # Tokenize the input audio
        inputs = processor(audio_chunk, return_tensors="pt", sampling_rate=rate)

        # Perform inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        # Decode the logits to text with character offsets
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, output_char_offsets=True, clean_up_tokenization_spaces=True)

        # Map the phonemes to visemes with offsets
        chunk_visemes_with_offsets = map_phonemes_to_visemes_with_offsets(transcription)
        visemes_with_offsets.extend(chunk_visemes_with_offsets)
    
    return visemes_with_offsets

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Audio recorder
audio_data = st.audio("Record an audio file", type=["wav"])

if uploaded_file is not None or audio_data is not None:
    if uploaded_file is not None:
        audio_file = uploaded_file
    else:
        audio_file = audio_data

    st.markdown('<div class="audio">', unsafe_allow_html=True)
    st.audio(audio_file, format='audio/wav')
    st.markdown('</div>', unsafe_allow_html=True)

    # Load and preprocess the audio file
    audio, rate = librosa.load(audio_file, sr=16000)

    # Start time measurement
    start = time.time()

    # Process audio in chunks if longer than 5 seconds
    if len(audio) / rate > 5:
        viseme_transcription_with_offsets = process_audio_chunks(audio, rate, chunk_duration=5)
    else:
        # Tokenize the input audio
        inputs = processor(audio, return_tensors="pt", sampling_rate=rate)

        # Perform inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        # Decode the logits to text with character offsets
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, output_char_offsets=True, clean_up_tokenization_spaces=True)

        # Map the phonemes to visemes with offsets
        viseme_transcription_with_offsets = map_phonemes_to_visemes_with_offsets(transcription)

    # End time measurement
    end = time.time()

    # Calculate time taken in milliseconds
    time_taken_ms = (end - start) * 1000

    # Print the time taken and transcription
    st.markdown('<div class="time-taken">Time Taken: {:.2f} ms</div>'.format(time_taken_ms), unsafe_allow_html=True)
    st.markdown('<div class="transcription">Transcription:</div>', unsafe_allow_html=True)
    st.write(transcription)

    # Display the viseme transcription with offsets
    st.markdown('<div class="json-output">Viseme Transcription with Offsets:</div>', unsafe_allow_html=True)
    st.json(viseme_transcription_with_offsets)
