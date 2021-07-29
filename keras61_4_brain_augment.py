# 훈련데이터를 기존데이터 20% 더 할것
# 성과비교
# temp에 저장 후 결과확인 뒤 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=False,   # 수평
    vertical_flip=False,
    zoom_range=1.2,
    fill_mode='nearest'     # 공백을 비슷한 이미지로 채우겠다
)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150), # 커지면 메모리가 터진다.
    batch_size=160,
    class_mode='binary',# train 폴더 안에 ad(이상스캔), normal(정상스캔)폴더 아래에 이미지가 들어있는데, 
    shuffle=True        # 이렇게 폴더 구조로 구분했을 경우 이게 라벨이 된다. 
)
# 실행하면 다음과 같은 메시지가 나온다. Found 160 images belonging to 2 classes. = 160장의 이미지를 찾았다.

xy_test = train_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150), # 커지면 메모리가 터진다.
    batch_size=120,
    class_mode='binary' # train 폴더 안에 ad(이상스캔), normal(정상스캔)폴더 아래에 이미지가 들어있는데, 
                        # 이렇게 폴더 구조로 구분했을 경우 이게 라벨이 된다. 
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

argument_size = int(160*0.2)
print(x_train.shape)
randidx = np.random.randint(x_train.shape[0], size=argument_size)

x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

x_argumented = x_argumented.reshape(x_argumented.shape[0], 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)

x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size),
                                batch_size=argument_size, shuffle=False,
                                save_to_dir='d:/temp/'
                                ).next()[0]

# x_train = np.concatenate((x_argumented, x_train))
# y_train = np.concatenate((y_argumented, y_train))

print('x_train shape',x_train.shape)

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

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# 증폭 전
# loss :  2.1861679553985596
# acc :  0.7083333134651184

# 증폭
# loss :  1.1037362813949585
# acc :  0.7833333611488342
