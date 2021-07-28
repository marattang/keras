from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0], type(x_train[0]))
print(y_train[0])

print(len(x_train[0]), len(x_train[1])) # 87, 56

# print(x_train[0].shape)

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print(type(x_train)) # <class 'numpy.ndarray'> list 형태로 들어가있기 때문에 shape가 가능하다

print('뉴스 기사의 최대길이 :', max(len(i) for i in x_train))  # 2376
# print('뉴스 기사의 최대길이 :', max(len(x_train)) 이건 안됨.
print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train)) # 145.5

plt.hist([len(s) for s in x_train], bins=50)
plt.show()

x_train = pad_sequences(x_train, maxlen=100, padding='pre') # (8892, 100)
x_test = pad_sequences(x_test, maxlen=100, padding='pre')   # (2246, 100)
print(x_train.shape, x_test.shape)
print(type(x_train), type(x_train[0]))
print(x_train[1]) # 100개에 데이터 13개가 붙었다.)


#  y 확인
print(np.unique(y_train)) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                          #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape) # (2246, 100) (2246, 46)

# 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=150))
model.add(LSTM(50, activation='relu'))
model.add(Dense(72))
model.add(Dense(46, activation='softmax'))
model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print('x_train', x_train.shape)
print('y_train', y_train.shape)
model.fit(x_train, y_train, epochs=100, batch_size=512)

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# acc :  0.5244879722595215