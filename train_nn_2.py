import os
import pickle
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Bidirectional

def char_to_onehot(char, all_chars):
    if char not in all_chars and char != "empty":
        raise ValueError("symbol not in character list")
    return [1.0 if char == c else 0.0 for c in all_chars]

def onehot_to_char(vector, all_chars):
    if 1.0 in vector:
        return all_chars[vector.index(1.0)]
    return "empty"

    
max_seq_len = 256
training_size = 0.9
unit_n = 512
epochs = 20
batch_size = 32
loss = 'categorical_crossentropy'
    
    
with open(r"c:\users\nh5\documents\g2p\corpres_data_short2_simple.pkl", "rb") as f:
    all_ortho_chars, all_trans_chars, data = pickle.load(f)
    
# all_ortho_chars.append("empty")
input_len = len(all_ortho_chars)
# all_trans_chars.append("empty")
output_len = len(all_trans_chars)
random.shuffle(data)
data_x = []
data_y = []
print("processing data")
for i, el in enumerate(data):
    print(f"{i}/{len(data)}", end="\r")
    x = [char_to_onehot(c, all_ortho_chars) for c in el[0]]
    y = [char_to_onehot(c, all_trans_chars) for c in el[1]]
    
    # print("was", len(x))
    if max_seq_len > len(x):
        x += [char_to_onehot("empty", all_ortho_chars) for i in range(max_seq_len - len(x))]
        # print("added")
    elif max_seq_len < len(x):
        x = x[:max_seq_len]
        # print("trimmed")
        
    if max_seq_len > len(y):    
        y += [char_to_onehot("empty", all_trans_chars) for i in range(max_seq_len - len(y))]
    elif max_seq_len < len(y):
        y = y[:max_seq_len]
    
    data_x.append(np.asarray(x))
    data_y.append(np.asarray(y))
print("converting to ndarray")
data_x = np.asarray(data_x)
data_y = np.asarray(data_y)

data_x, data_x_test = data_x[:int(len(data_x) * training_size)], data_x[int(len(data_x) * training_size):]
data_y, data_y_test = data_y[:int(len(data_y) * training_size)], data_y[int(len(data_y) * training_size):]

model = Sequential()
model.add(Bidirectional(LSTM(unit_n, return_sequences=True, input_shape=(max_seq_len, input_len))))
# model.add(Bidirectional(LSTM(unit_n, return_sequences=True, input_shape=(max_seq_len, input_len))))
# model.add(Bidirectional(LSTM(unit_n, return_sequences=True, input_shape=(max_seq_len, input_len))))
model.add(Dense(output_len, activation="softmax"))
model.compile(loss=loss, optimizer="Adam", metrics=["categorical_accuracy"])

history = model.fit(
    data_x,
    data_y,
    epochs=epochs,
    validation_split=0.1,
    batch_size=batch_size)

model.save(os.path.join(r"c:\users\nh5\documents\g2p", "corpres_90_20epochs"))

print("testing")

    
test_predict = model.predict(data_x_test)
acc = 0
for x, y in zip(data_y_test, test_predict):
    for xs, ys in zip(x, y):
        ys = np.round(ys)
        if all([xs[i] == ys[i] for i in range(len(ys))]):
            acc += 1
acc /= (data_y_test.shape[0] * data_y_test.shape[1])
print(acc)

spaces = True

print("test yourself...")
while True:
    x_test = input()
    if spaces:
        x_test = x_test.split(" ")
    else:
        x_test = list(x_test)
        new_x_test = []
        for i in x_test:
            if i in "01'":
                new_x_test[-1] += i
            else:
                new_x_test.append(i)
        x_test = new_x_test
    x_test += ["empty" for i in range(max_seq_len - len(x_test))]
    try:
        x_test = np.asarray([[char_to_onehot(i, all_ortho_chars) for i in x_test]])
    except ValueError:
        print("incorrect input")
        continue
    res = np.round(model.predict(x_test))
    res = [onehot_to_char(list(i), all_trans_chars) for i in res[0]]
    res = " ".join([i for i in res if i != "empty"])
    print(res)

