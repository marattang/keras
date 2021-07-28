import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,   # 수평
    vertical_flip=True,
    width_shift_range=0.1,  # 원래 이미지에서 좌우로 0.1정도 이동 가능
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'     # 공백을 비슷한 이미지로 채우겠다
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150), # 커지면 메모리가 터진다.
    batch_size=5,
    class_mode='binary',# train 폴더 안에 ad(이상스캔), normal(정상스캔)폴더 아래에 이미지가 들어있는데, 
    shuffle=True        # 이렇게 폴더 구조로 구분했을 경우 이게 라벨이 된다. 
)
# 실행하면 다음과 같은 메시지가 나온다. Found 160 images belonging to 2 classes. = 160장의 이미지를 찾았다.

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150), # 커지면 메모리가 터진다.
    batch_size=5,
    class_mode='binary' # train 폴더 안에 ad(이상스캔), normal(정상스캔)폴더 아래에 이미지가 들어있는데, 
                        # 이렇게 폴더 구조로 구분했을 경우 이게 라벨이 된다. 
)
# Found 120 images belonging to 2 classes. 테스트는 120장

# print(xy_train)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022A82218550>
# print(xy_train[0]) # 라벨값이 나온다. y는 batch size.
# print(xy_train[0][0]) # x값
# print(xy_train[0][1]) # y값
# # print(xy_train[0][2]) # 없음

# print(xy_train[0][0].shape, xy_train[0][1].shape) # (5, 150, 150, 3) (5,) => (batch size, target size, color) (batch size)
#                                                   # 총 몇 장 이미지가 있는지 폴더 안에서 확인할 수 있음. 총 이미지는 160장 
#                                                   # binary기 때문에 ad의 이미지는 0혹은 1 라벨링 normal은 나머지 0 1중 하나로 라벨링 된다.
#                                                   # batch size를 5로 조절했기 때문에 나머지 xy_train[1] ~ [31]까지 5개씩 딱딱 떨어지게 된다.  

# print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

# 2. 모델
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3), activation='relu')) # 흑백 데이터지만 자동으로 컬러로 인식한다.
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(인수 두 개를 받아야 하기 때문에 지금은 x데이터, y데이터가 묶여져 있어서 fit을 사용할 순 없음.)
hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32,
                            validation_data=xy_test,
                            validation_steps=4
                            )

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할 것
print('acc :', acc[-1])
print('val_acc : ', val_acc[-1])
