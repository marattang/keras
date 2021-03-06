import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)

# 2. 모델구성
# 가장 먼저 맨 윗 층 레이어에 Bidirectional 
model = Sequential()
# model.add(SimpleRNN(units=20, activation='relu', input_shape=(3, 1)))
# model.add(LSTM(units=20, activation='relu', input_shape=(3, 1)))
# model.add(LSTM(units=20, activation='relu', input_shape=(3, 1)))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(10, activation='relu'))
# input shape가 있으면 맞춰줘야 하기 때문에 에러가 난다?
# 전달받은 노드만 할 경우
# 0을 패딩으로 채웠을 경우 반대방향으로 학습을 진행했을 때 안 좋아질 수도 있다.
model.add(LSTM(10, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(Bidirectional(LSTM(10, activation='relu')))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 20)                1760
_________________________________________________________________
dense (Dense)                (None, 15)                315
_________________________________________________________________
dense_1 (Dense)              (None, 10)                160
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11
=================================================================
''' 
# LSTM PARAM은 연산을 4번하기 때문에 RNN PARAM * 4가 된다.
# (Input + bias) * output + output * output= (Input + bias + output) * output
# 이전 layer의 가중치가 반영되기 때문에 식이 완성된다.
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1200) 

# # # 4. 평가, 예측
x_input = np.array([5, 6, 7]).reshape(1,3,1)
result = model.predict(x_input)
print(result)

# [[8.]]

model.summary()
