from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Embedding

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=100
)

# 실습 시작!!
print(x_train[0], type(x_train[0]))
print(y_train[0])

print(len(x_train[0]), len(x_train[1]))

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(type(x_train))

print('최대 길이 :', max(len(i) for i in x_train)) # 2494
print('평균 길이 :', sum(map(len, x_train))/ len(x_train)) # 238.71364

# plt.hist([len(s) for s in x_test], bins=50)
# plt.show()

x_train = pad_sequences(x_train, maxlen=200, padding='pre') # (25000, 200)
x_test = pad_sequences(x_test, maxlen=200, padding='pre') # (25000, 200)
print(x_train.shape, x_test.shape)
# 모델

print(np.unique(y_train))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=518))
model.add(LSTM(50, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_acc', patience=10, mode='max')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=1024, callbacks=[es], validation_split=0.15)

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# acc:  0.6212000250816345