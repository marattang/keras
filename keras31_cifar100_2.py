# 완벽한 모델 구성import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool2D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout
import numpy as np
import time
import matplotlib.pyplot as plt

# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


# 전처리
x_train = x_train.reshape(50000, 32 * 32 * 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 32 * 32 * 3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(50000, 1)
y_test = y_test.reshape(10000, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

# 2. 모델링
model = Sequential()
model.add(Conv2D(filters=128, activation='relu', kernel_size=(2,2), padding='valid',  input_shape=(32, 32, 3)))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(128, (2,2), activation='relu', padding='valid'))
model.add(MaxPool2D())
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Flatten())                              
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(mode='auto', monitor='val_loss', patience=4)
start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=128, validation_split=0.02, callbacks=[es])
end = time.time() - start

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()

print("걸린시간 : ", end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])


# 
# acc :  0.4268999993801117

# Standard
# acc :  0.4302000105381012

# Maxabs
# acc :  0.41819998621940613

# RobustScaler
# acc :  0.4251999855041504

# 모델 수정 = 노드 감소 + same -> padding 속성 일부 변경
# StandardScaler
# acc :  0.4408999979496002
# RobustScaler
# acc :  0.4345000088214874
# MinMax
# acc :  0.42179998755455017
# PowerTransformer
# acc :  0.4287000000476837
# QuantileTransformer
# acc :  0.4235000014305115