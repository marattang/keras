import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input

# 1. 데이터
# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])


print(x.shape, y.shape) # (4, 3) (4,)

# x = x.reshape(13, 3, 1) # (batch_size, timesteps, feature)
x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1, x_predict.shape[0], 1)
print(x_predict.shape)
# 2. 모델구성
input = Input(shape=(3,1))
dense1 = SimpleRNN(units=32, activation='relu', input_shape=(3, 1))(input)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
dense4 = Dense(16, activation='relu')(dense3)
dense5 = Dense(8, activation='relu')(dense4)
output = Dense(1, activation='relu')(dense5)

model = Model(inputs=input, outputs=output)
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
model.fit(x, y, batch_size=1, epochs=250) 

# # 4. 평가, 예측
x_input = np.array([5, 6, 7]).reshape(1,3,1)
result = model.predict(x_input)
print(result)

# [[8.080044]]