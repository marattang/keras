# 함수형 모델의 레이어명에 대한 고찰!! xx
import numpy as np
from numpy.core.fromnumeric import transpose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101),
            range(100), range(401, 501)])

x = np.transpose(x)
print(x.shape)  # (100, 5)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape)  # (100, 2)

#2. 모델
input1 = Input(shape=(5,))
xx = Dense(3)(input1)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx)

model = Model(inputs=input1, outputs=output1)

model.summary()


# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

#3. 컴파일, 훈련
#4. 평가, 예측
