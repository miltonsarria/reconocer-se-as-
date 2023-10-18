import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from pathlib import Path


DATA_PATH = os.path.join('MP_Data')
#raw_leonardo
#raw_miguel_angie
#MP_Data
#Actions
actions = np.array(['hola','gracias','TeQuiero'])
#30 videos worth of data
no_sequences = 50
#30 frames
sequence_length = 30
#cargar datos y asignar etiquetas
sequences, labels = [], []
label_map = {label:num for num, label in enumerate(actions)}
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH,action, str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
#crear matriz de datos de entrada X y vector de respuestas Y
X = np.array(sequences)
y = to_categorical(labels).astype(int)

## cargar modelo
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

model = Sequential()
model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128,return_sequences=True, activation = 'relu'))
model.add(LSTM(64, return_sequences = False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(actions.shape[0],activation='softmax'))

res = [.2,0.7,.01]

actions[np.argmax(res)]
model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
actions[np.argmax(res[1])]

model.load_weights('ModeloSenas.h5')

y_hat = model.predict(X)

y_true = np.argmax(y, axis=1).tolist()
y_hat = np.argmax(y_hat, axis=1).tolist()
mc = multilabel_confusion_matrix(y_true, y_hat)
ac = accuracy_score(y_true, y_hat)
print(mc)
print("ac", ac)
