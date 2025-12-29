import streamlit as st
import numpy as np
import librosa as lib
import tempfile
import tensorflow as tf
from tensorflow import keras
import os
from pydub import AudioSegment
import requests


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MODEL_URL = "https://github.com/Sageioi/Eureka/releases/download/v1.0.0/AudioModel.keras"
MODEL_PATH = "AudioModel.keras"

st.title("AudioSoul -- From the creator of Dermatect")
st.markdown(":red[_______________________________________________________________________________________]")
st.header("A product of Eureka AI laboratory formerly known as RetinAI Computer Vision Laboratory")
st.text(
    "Upload a short recording in wav/mp3/aac format and await a secret message only you can unlock.\n"
    "This product is an initiative of Eureka AI laboratory for AI for therapy."
)


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL, allow_redirects=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    return keras.models.load_model(MODEL_PATH)

model = load_model()


def preprocess(wav_tensor):
    if not isinstance(wav_tensor, tf.Tensor):
        print("Input to preprocess is not a tensor.")
        return None

    TARGET_LENGTH = 48000
    wav_tensor = wav_tensor[:TARGET_LENGTH]
    padding_needed = TARGET_LENGTH - tf.shape(wav_tensor)[0]
    zero_padding = tf.zeros([padding_needed], dtype=tf.float32)
    wav_tensor = tf.concat([zero_padding, wav_tensor], 0)

    spectrogram = tf.signal.stft(
        wav_tensor,
        frame_length=320,
        frame_step=32
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# --------------------- USER INPUT ----------------------
name = st.text_input(label="What's your name")
if name.strip() == "":
    name = "User"

uploaded_file = st.file_uploader(
    f"Drop a piece of your mind here, {name}",
    type=['wav', 'mp3', 'aac'],
    key="audio",
    help="Supported formats: WAV, MP3, AAC."
)


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name
    temp_wav_path = temp_input_path + ".wav"

    try:
    
        audio = AudioSegment.from_file(temp_input_path)
        audio.export(temp_wav_path, format="wav")


        y_np, sr = lib.load(temp_wav_path, sr=16000, mono=True)
        wav_tensor = tf.convert_to_tensor(y_np, dtype=tf.float32)
        spectrogram_3d = preprocess(wav_tensor)

        if spectrogram_3d is not None:
          
            spectrogram_4d = tf.expand_dims(spectrogram_3d, axis=0)
            predictions = model.predict(spectrogram_4d)
            y_pred_index = np.argmax(predictions)
            st.success(f"Dear {name}, I dropped in a secret message for you.")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.warning("Nothing for you.")
        y_pred_index = None

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)


    st.markdown(":red[_______________________________________________________________________________________]")
    with st.expander(label="Secret Message"):
        messages = {
            0: [f"{name}, You are in a right frame of mind", f"I am happy for you, {name}", f"You are doing well, {name}"],
            1: [f"{name}, You have a happy soul", f"Keep soaring, {name}", f"You sound great, {name}"],
            2: [f"{name}, It is well with your soul", f"Everything will be alright, {name}", f"You will be better, {name}"],
            3: [f"{name}, Something great is coming your way", f"The stars will align, {name}", f"I am just an AI model designed for therapy, {name}"]
        }

        if y_pred_index in messages:
            number = np.random.randint(0, len(messages[y_pred_index]))
            st.text(f"Message for {name}: {messages[y_pred_index][number]}")
        else:
            st.text("No message available. Please upload a clear audio file.")
