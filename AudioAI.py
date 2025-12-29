import os
import tensorflow as tf, librosa as lib, matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




data_path = r"C:\Users\USER\OneDrive\Desktop\AudioAI\train"
emotion = []
emotions = {"joyfully": [], "euphoric": [], "sad": [], "surprised": []}
emotions_wav = {"joyfully": [], "euphoric": [], "sad": [], "surprised": []}


for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                emotion.append(file_path)


print(f"Total files found: {len(emotion)}\n")

for file_path in emotion:
    filename = os.path.basename(file_path)
    if filename == "joyfully.wav" or filename == "joyfully.amr" or filename == "joyfully.mpeg":
        emotions["joyfully"].append(file_path)
    elif filename == "euphoric.wav" or filename == "euphoric.amr"or filename == "euphoric.mpeg":
        emotions["euphoric"].append(file_path)
    elif filename == "sad.wav" or filename == "sad.amr" or filename == "sad.mpeg":
        emotions["sad"].append(file_path)
    elif filename == "surprised.wav" or filename == "surprised.amr"or filename =="surprised.mpeg":
        emotions["surprised"].append(file_path)
    else:
        print(f"Unmatched file: {filename} (Path: {file_path})")


def load_audio(filepath):
    try:
        wav, sample_rate = lib.load(filepath,sr=16000,mono=True)
        wav = tf.convert_to_tensor(wav , dtype = tf.float32)
    except:
        return "This file is either a wrong file or an audio file whose format is not supported"
    return wav

for joy in emotions["joyfully"]:
    joy_wav = load_audio(joy)
    emotions_wav["joyfully"].append(joy_wav)
for euph in emotions["euphoric"]:
    euph_wav = load_audio(euph)
    emotions_wav["euphoric"].append(euph_wav)

for sad in emotions["sad"]:
    sad_wav = load_audio(sad)
    emotions_wav["sad"].append(sad_wav)

for surp in emotions["surprised"]:
    surp_wav = load_audio(surp)
    emotions_wav["surprised"].append(surp_wav)



def preprocess(file_path,label):
    wav = load_audio(file_path)
    if isinstance(wav,(str,dict)):
        print("File unsupported for AI training.")
        return None, label
    wav = wav[:48000]
    zero_padding =  tf.zeros([48000]-tf.shape(wav),dtype = tf.float32)
    wav = tf.concat([zero_padding,wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320,frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis = 2)
    return spectrogram,label

label_map = {
    "joyfully": 0,
    "euphoric": 1,
    "sad": 2,
    "surprised": 3
}

data = {"x_train":[],"y_train":[]}

for emotion_name, file_list in emotions.items():
    for file_path in file_list:
        label = label_map[emotion_name]
        spectrogram, output_label = preprocess(file_path, label)
        if spectrogram is None:
            continue
        data["x_train"].append(spectrogram)
        data["y_train"].append(output_label)

x_train = np.array(data["x_train"])
y_train = np.array(data["y_train"])


test_path = r"C:\Users\USER\OneDrive\Desktop\AudioAI\test"
emotion_test = []
emotions_test = {"joyfully": [], "euphoric": [], "sad": [], "surprised": []}
emotions_wav_test = {"joyfully": [], "euphoric": [], "sad": [], "surprised": []}


for folder in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder)
    
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                emotion_test.append(file_path)


print(f"Total files found: {len(emotion)}\n")

for file_path in emotion:
    filename = os.path.basename(file_path)
    if filename == "joyfully.wav" or filename == "joyfully.amr" or filename == "joyfully.mpeg":
        emotions_test["joyfully"].append(file_path)
    elif filename == "euphoric.wav" or filename == "euphoric.amr"or filename == "euphoric.mpeg":
        emotions_test["euphoric"].append(file_path)
    elif filename == "sad.wav" or filename == "sad.amr" or filename == "sad.mpeg":
        emotions_test["sad"].append(file_path)
    elif filename == "surprised.wav" or filename == "surprised.amr"or filename =="surprised.mpeg":
        emotions_test["surprised"].append(file_path)
    else:
        print(f"Unmatched file: {filename} (Path: {file_path})")


def load_audio(filepath):
    try:
        wav, sample_rate = lib.load(filepath,sr=16000,mono=True)
        wav = tf.convert_to_tensor(wav , dtype = tf.float32)
    except:
        return "This file is either a wrong file or an audio file whose format is not supported"
    return wav

for joy in emotions["joyfully"]:
    joy_wav = load_audio(joy)
    emotions_wav_test["joyfully"].append(joy_wav)
for euph in emotions["euphoric"]:
    euph_wav = load_audio(euph)
    emotions_wav_test["euphoric"].append(euph_wav)

for sad in emotions["sad"]:
    sad_wav = load_audio(sad)
    emotions_wav_test["sad"].append(sad_wav)

for surp in emotions["surprised"]:
    surp_wav = load_audio(surp)
    emotions_wav_test["surprised"].append(surp_wav)



label_map = {
    "joyfully": 0,
    "euphoric": 1,
    "sad": 2,
    "surprised": 3
}

data_test = {"x_test":[],"y_test":[]}

for emotion_name, file_list in emotions.items():
    for file_path in file_list:
        label = label_map[emotion_name]
        spectrogram, output_label = preprocess(file_path, label)
        if spectrogram is None:
            continue
        data_test["x_test"].append(spectrogram)
        data_test["y_test"].append(output_label)
        

x_test = np.array(data_test["x_test"])
y_test = np.array(data_test["y_test"])


model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(1491, 257, 1)),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(16, activation="relu"), 
    keras.layers.Dense(4, activation="softmax"),
])
model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_train,y_train, batch_size = 5, epochs = 9)
model.evaluate(x_test,y_test,batch_size = 5)
model.save("AudioModel.keras")

