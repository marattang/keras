# 훈련데이터를 기존데이터 20% 더 할것
# 성과비교
# temp에 저장 후 결과확인 뒤 삭제
# 실습
# 약속 잘 지키기 왕
# 말과 사람 데이터셋 완성
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Flatten
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import math
# '''
train_datagen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip=False,
    horizontal_flip=False,
    zoom_range=0.1)

xy = train_datagen.flow_from_directory(
    '../_data/horse-or-human',
    target_size=(100, 100),
    batch_size=1027,
    class_mode='sparse',
    shuffle=True
)

x_train, x_test, y_train, y_test = train_test_split(xy[0][0], xy[0][1], train_size=0.7, random_state=10, shuffle=True)

argument_size = math.ceil(x_train.shape[0] * 0.2)
print(argument_size)
print(type(argument_size))

randidx = np.random.randint(x_train.shape[0], size=argument_size)

x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size),
                                batch_size=argument_size, shuffle=False,
                                save_to_dir='d:/temp/'
                                ).next()[0]

x_train = np.concatenate((x_argumented, x_train))                                
y_train = np.concatenate((y_argumented, y_train))
# '''

# 학습
# '''

print('x_train shape',x_train.shape)

model = Sequential()
model.add(Conv2D(64, input_shape=(100, 100, 3), kernel_size=(2, 2), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='min')
model.fit(x_train, y_train, validation_split=0.1, epochs=150, callbacks=[es])

result = model.evaluate(x_train, y_train)
print('loss :', result[0])
print('acc :', result[1])
# '''
# size 250, 250
# loss : 0.056782789528369904
# acc : 0.9935064911842346

# size 100, 100
# loss : 0.001572958892211318
# acc : 1.0

# 증폭 후 
# loss : 0.1141144260764122
# acc : 0.9547563791275024