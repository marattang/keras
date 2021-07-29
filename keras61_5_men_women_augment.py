# 훈련데이터를 기존데이터 20% 더 할것
# 성과비교
# temp에 저장 후 결과확인 뒤 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, GlobalAveragePooling2D, Dense, Dropout, MaxPool2D, LSTM, SimpleRNN, Conv1D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math
# 실습
# men women 데이터로 모델링 구성하기
# 문제 1. 데이터 용량이 큼.

train_datagen = ImageDataGenerator(
    rescale=1./120,
    vertical_flip=False,
    horizontal_flip=False,
    zoom_range=0.2
  )

start = time.time()

xy = train_datagen.flow_from_directory(
    '../_data/men_women',
    target_size=(80, 80),
    batch_size=3309,
    class_mode='binary',
    shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(xy[0][0], xy[0][1], train_size=0.7, shuffle=True, random_state=10)
argument_size = math.ceil(x_train.shape[0]*0.2)

randidx = np.random.randint(x_train.shape[0], size=argument_size)

x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

x_argumented = x_argumented.reshape(x_argumented.shape[0], 80, 80, 3)
x_train = x_train.reshape(x_train.shape[0], 80, 80, 3)

end = time.time() - start
# print('긁어오는 데 걸린 시간 :', end) 0.09275674819946289

print('xy : ',xy)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022A82218550>
# print('xy[0] : ',xy[0]) # 라벨값이 나온다. y는 batch size.
# print('xy[0][0] : ',xy[0][0]) # x값
# print('xy[0][1] : ',xy[0][1]) # y값
# # print(xy_train[0][2]) # 없음
# print(xy[0][0].shape, xy[0][1].shape) # (3309, 250, 250, 3) (3309,)

x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size),
                                batch_size=argument_size, shuffle=False,
                                save_to_dir='d:/temp/'
                                ).next()[0]

x_train = np.concatenate((x_argumented, x_train))
y_train = np.concatenate((y_argumented, y_train))

# 학습
# '''

# 본인 사진으로 predict 하기. d:\data 안에 사진 넣고

# 모델 1

print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)

model = Sequential()
model.add(Conv2D(64, input_shape=(80, 80, 3), kernel_size=(8,8), activation='relu'))
model.add(Dropout(0.8))
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# classification은 loss말고 acc가 더 중요한거같음.
model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(0.0005), metrics=['acc'])

learning_start = time.time()

es = EarlyStopping(monitor='val_acc', mode='max', patience=15)
# model.fit(x_train, y_train, epochs=50, validation_split=0.05, batch_size=8, callbacks=[es])
model.fit(x_train, y_train, epochs=50, validation_split=0.05, batch_size=126, callbacks=[es])
learning_end = (time.time() - learning_start)/60
print('학습 걸린 시간(분) : ', learning_end)

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])
# '''

# Conv2d
# y_predict : [[0.6189552 ]
#  [0.03671281]
#  [0.84080505]
#  [0.5594377 ]
#  [0.45257118]]

# loss :  0.602618932723999
# acc :  0.6928499341011047

'''
0 번째 : [75.04062]의 확률로 남자
1 번째 : [25.814837]의 확률로 남자
2 번째 : [24.355686]의 확률로 남자
3 번째 : [97.525116]의 확률로 남자
4 번째 : [62.31045]의 확률로 남자
'''
# 0.7까지 올리기
# 데이터 증폭 후
# loss :  1.1295862197875977
# acc :  0.6334340572357178