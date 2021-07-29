# 훈련데이터를 10만개로 증폭
# 성과비교
# temp에 저장 후 결과확인 뒤 삭제

# overfit 극복~~
# 1. 전체 훈련 데이터를 늘린다. 많이 많이 train data가 많으면 많을 수록 오버핏이 준다. -> 실질적으로 불가능한 경우가 많다.
#  - 증폭시킬려고 해도 비슷한 양으로 증폭되기 때문에 한계가 있다.
# 2. Normalization : 정규화
#  - Regulization, Standardzation이랑 헷갈리지 말기. layer값에서 다음 값으로 전달해 줄 때 activation으로 값을 감싸서 다음 layer로 전달해주게 되는데
#  - 그 값 자체도 Normalize 하지 않다는 얘기다. layer별로도 Normalize해주는 게 어떠냐는 얘기?
# 3. dropout
#  - 전체적으로 연결되어있는 레이어의 구성을 Fully Connected layer라고 하는데, 
# 완벽한 모델 구성import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool2D, GlobalAveragePooling1D, LSTM, GlobalAveragePooling2D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.python.keras.layers.core import Dropout
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.2,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'
)

argument_size = 50000
randidx = np.random.randint(x_train.shape[0], size=argument_size)

x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

x_argumented = x_argumented.reshape(x_argumented.shape[0], 32, 32, 3)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size),
                                    batch_size=argument_size, shuffle=False,
                                    save_to_dir='d:/temp/'
                                    ).next()[0]

# x_train = np.concatenate((x_argumented, x_train))
# y_train = np.concatenate((y_argumented, y_train))

# 전처리
# 데이터의 내용물과 순서가 바뀌면 안된다.

# x_train = x_train.reshape(50000, 32, 32, 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
# x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

# 2. 모델링
# RNN
# model = Sequential()
# model.add(LSTM(8, input_shape=(32*32,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))



# DNN
# model = Sequential()
# model.add(Dense(528, input_shape=(32*32,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(528, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))


# CNN
model = Sequential()
model.add(Conv2D(filters=128, activation='relu', kernel_size=(2,2), padding='valid',  input_shape=(32, 32, 3)))
model.add(MaxPool2D())
model.add(Dropout(0.8))
model.add(GlobalAveragePooling2D())      
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(mode='auto', monitor='val_loss', patience=5)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_9_MCP.hdf', save_best_only=True)
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=512, validation_split=0.05, verbose=1, callbacks=[es, cp])
# model.save('./_save/ModelCheckPoint/keras48_9_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_9_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_9_MCP.hdf')

end = time.time() - start

print("걸린시간 : ", end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

# check point
# loss :  2.0895752906799316
# acc :  0.45660001039505005

# 단순 비교를 위해 기존 모델로는 시간이 오래걸려서 layer, node수 감소
# 증폭 전


# 증폭 후
# loss :  3.4414620399475098
# acc :  0.1956000030040741

# 증폭 전
# loss :  3.341580867767334
# acc :  0.21279999613761902