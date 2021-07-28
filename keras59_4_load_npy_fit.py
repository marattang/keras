import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

# 53_3 npy 이용해서 모델 완성

x_train = np.load('./_save/_npy/k59_3_x_train.npy')
y_train = np.load('./_save/_npy/k59_3_y_train.npy')
x_test = np.load('./_save/_npy/k59_3_x_test.npy')
y_test = np.load('./_save/_npy/k59_3_y_test.npy')


model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3), activation='relu')) # 흑백 데이터지만 자동으로 컬러로 인식한다.
model.add(Dropout(0.8))
model.add(Conv2D(16, (2, 2), activation='relu')) # 흑백 데이터지만 자동으로 컬러로 인식한다.
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(인수 두 개를 받아야 하기 때문에 지금은 x데이터, y데이터가 묶여져 있어서 fit을 사용할 순 없음.)
# hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32,
#                             validation_data=xy_test,
#                             validation_steps=4
#                             )
hist = model.fit(x_train, y_train, epochs=90, steps_per_epoch=32,
                validation_split=0.1, batch_size=5, validation_steps=4)

