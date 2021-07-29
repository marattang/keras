# 실습
# categorical_crossentropy와 sigmoid 조합
# 훈련데이터를 기존데이터 20% 더 할것
# 성과비교
# temp에 저장 후 결과확인 뒤 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model, load_model, Sequential
import time
import tensorflow as tf
import math

# '''
train_datagen = ImageDataGenerator(
    rescale=1./120,
    vertical_flip=False,
    horizontal_flip=False,
    zoom_range=0.1
)

start = time.time()
test_datagen = ImageDataGenerator(rescale=1./120,)

train = train_datagen.flow_from_directory(
    '../_data/cat_and_dog/training_set',
    target_size=(80, 80),
    batch_size=8005,
    class_mode='binary',
    shuffle=False
)

test = train_datagen.flow_from_directory(
    '../_data/cat_and_dog/test_set',
    target_size=(80, 80),
    batch_size=2025,
    class_mode='binary',
    shuffle=False
)
x_train = train[0][0]
y_train = train[0][1]

x_test = test[0][0]
y_test = test[0][1]

argument_size = math.ceil(x_train.shape[0] * 0.2)

randidx = np.random.randint(x_train.shape[0], size=argument_size)

x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

x_argumented = x_argumented.reshape(x_argumented.shape[0], 80, 80, 3)
x_train = x_train.reshape(x_train.shape[0], 80, 80, 3)
x_test = x_test.reshape(x_test.shape[0], 80, 80, 3)

x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size),
                                    batch_size=argument_size, shuffle=False,
                                    save_to_dir = 'd:/temp/'
                                    ).next()[0]

x_train = np.concatenate((x_argumented, x_train))
y_train = np.concatenate((y_argumented, y_train))
# '''
# 학습
# '''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(36, input_shape=(80, 80, 3), kernel_size=(5,5), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(2, activation='softmax'))


print('x_train shape', x_train.shape)
es = EarlyStopping(monitor='val_loss', mode='auto', patience=8)
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(0.0005), metrics=['acc'])
model.fit(x_train, y_train, batch_size=128, epochs=150, callbacks=[es], validation_split=0.15)

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# '''
# activation = sigmoid
# loss :  0.8603302836418152
# acc :  0.686109721660614

# activation = softmax
# loss :  0.7676746249198914
# acc :  0.6875926852226257 

# hypter parameter tuning batch size 증가->
# loss :  0.6418152451515198
# acc :  0.6747404932975769

# 증폭 후
# loss :  0.7295843958854675
# acc :  0.6658428311347961