import streamlit as st
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

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
        This application allows you to upload an audio file, which will be processed to generate a viseme transcription with offsets.
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
    "m": "B_M_P"
}

def map_phonemes_to_visemes_with_offsets(transcription):
    visemes_with_offsets = []
    for item in transcription['char_offsets'][0]:
        phoneme = item['char']
        start_offset = item['start_offset']
        end_offset = item['end_offset']
        viseme = phonemeToVisemeMapping.get(phoneme, "SIL")  # Default to "SIL" if phoneme not found
        visemes_with_offsets.append({
            'viseme': viseme,
            'start_offset': start_offset,
            'end_offset': end_offset
        })
    return visemes_with_offsets

# File uploader for audio file
#st.markdown('<div class="upload"><input type="file" id="file-upload" accept=".wav,.mp3"></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.markdown('<div class="audio">', unsafe_allow_html=True)
    st.audio(uploaded_file, format='audio/wav')
    st.markdown('</div>', unsafe_allow_html=True)

    # Load and preprocess the audio file
    audio, rate = librosa.load(uploaded_file, sr=16000)

    # Tokenize the input audio
    inputs = processor(audio, return_tensors="pt", sampling_rate=rate)

    # Start time measurement
    start = time.time()

    # Perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Decode the logits to text with character offsets
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, output_char_offsets=True, clean_up_tokenization_spaces=True)

    # End time measurement
    end = time.time()

    # Calculate time taken in milliseconds
    time_taken_ms = (end - start) * 1000

    # Print the time taken and transcription
    st.markdown('<div class="time-taken">Time Taken: {:.2f} ms</div>'.format(time_taken_ms), unsafe_allow_html=True)
    st.markdown('<div class="transcription">Transcription:</div>', unsafe_allow_html=True)
    st.write(transcription)

    # Map the phonemes to visemes with offsets
    viseme_transcription_with_offsets = map_phonemes_to_visemes_with_offsets(transcription)

    # Display the viseme transcription with offsets
    st.markdown('<div class="json-output">Viseme Transcription with Offsets:</div>', unsafe_allow_html=True)
    st.json(viseme_transcription_with_offsets)
