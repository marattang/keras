from keras_preprocessing.image import image_data_generator
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling1D, LSTM
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
import time

from tensorflow.python.keras.layers.core import Dropout



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,   # 수평
    vertical_flip=False,
    width_shift_range=0.1,  # 원래 이미지에서 좌우로 0.1정도 이동 가능
    height_shift_range=0.2,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'     # 공백을 비슷한 이미지로 채우겠다
)
# train_datagen = ImageDataGenerator(rescale=1./255)

argument_size=40000

randidx = np.random.randint(x_train.shape[0], size=argument_size) # train data에서 40000개의 난수 생성
print(x_train.shape[0]) # 
print(randidx) # [ 2926  6407 14388 ... 30007 13703 27702] => 데이터들의 y값은 바뀌면 안되기 때문에 x데이터를 약간씩 수정한다.
print(randidx.shape) # (40000,)

x_argumented = x_train[randidx].copy() # copy안해도 상관없지만 메모리가 공유될 수도 있기 때문에 사용
y_argumented = y_train[randidx].copy()

# print(x_argumented.shape) # (40000, 28, 28)

x_argumented = x_argumented.reshape(x_argumented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size), # y argument 넣지 않은 이유 -> 어차피 증폭하더래도, 셔플을 하지 않았기 때문에 상관 없음
                                  batch_size=argument_size, shuffle=False).next()[0] # iterator의 0번째 => x만 쏙 빠진다.
# 4만장의 데이터 -> 4만장의 데이터로 바뀌고 순서가 바뀌지 않아서 순서는 그대로 유지되기 때문에 y자리에 y_argumented를 굳이 넣지 않아도 상관 없다.
print(x_argumented.shape) #(40000, 28, 28, 1)

# x_train = np.concatenate((x_argumented, x_train))
# y_train = np.concatenate((y_argumented, y_train))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

# 모델 완성
# 비교대상 증폭x fashion mnist


print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 2. 모델링
model = Sequential()

# CNN
model.add(Conv2D(filters=32, activation='relu', kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
model.add(Flatten())                                      # (N, 135)
model.add(Dropout(0.8))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(mode='min', monitor='val_loss', patience=5)
cp = ModelCheckpoint(monitor='val_accuracy', mode='max', filepath='./_save/ModelCheckPoint/keras48_7_MCP.hdf', save_best_only=True)
start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=512, validation_split=0.05, callbacks=[es, cp])
# model =load_model('./_save/ModelCheckPoint/keras48_7_model.h5')
model = load_model('./_save/ModelCheckPoint/keras48_7_MCP.hdf')
end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_7_model.h5')

print('걸린시간 : ', end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# CNN의 경우
# accuracy :  0.9075000286102295 제일 높음 나머지 0.9014, 0.1

# layer 수가 많아 줄임 -> 단순 acc 수치 비교
# 증폭
# loss :  0.37354516983032227
# accuracy :  0.8787000179290771

# 증폭x
# loss :  0.3159727454185486
# accuracy :  0.8891000151634216

# 