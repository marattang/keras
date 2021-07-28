import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. 데이터 2x10인걸 10행 2열로 바꿔야함.
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               [1, 1.1, 1.2, 1.3, 1.4, 1.5, 
                1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]])
# 행무시, 열 우선 input layer
x = np.transpose(x)
print(np.shape(x))
y = np.array( [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#  완성하시오

# 2. 모델 구성
model = Sequential()
model.add(Dense(9, input_dim=3))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=2500, batch_size=7)

# 4. 평가, 예측
x_pred = np.array([[10, 1.3, 1]]) # 3열이 나와야 하기 때문에 []
# x_pred2 = np.array([10, 1.3, 1]) 3행이 되기 때문에 안됨. 묶어줘야함.

loss = model.evaluate(x, y)
result = model.predict(x)

print('loss : ', loss)
print('10, 1.3, 1의 예측값 : ', result)
print(x[:,0])
# y, x의 차원 수가 맞지 않아서 산점도가 나오지 않음.
plt.scatter(x[:,0], y)
plt.plot(x[:,0], result, color='red')
plt.scatter(x[:,1], y)
plt.plot(x[:,1], result, color='red')
plt.scatter(x[:,2], y)
plt.plot(x[:,2], result, color='red')
plt.show()
