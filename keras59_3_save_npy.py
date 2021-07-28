import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    batch_size=160,
    class_mode='binary',# train 폴더 안에 ad(이상스캔), normal(정상스캔)폴더 아래에 이미지가 들어있는데, 
    shuffle=True        # 이렇게 폴더 구조로 구분했을 경우 이게 라벨이 된다. 
)
# 실행하면 다음과 같은 메시지가 나온다. Found 160 images belonging to 2 classes. = 160장의 이미지를 찾았다.

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150), # 커지면 메모리가 터진다.
    batch_size=120,
    class_mode='binary' # train 폴더 안에 ad(이상스캔), normal(정상스캔)폴더 아래에 이미지가 들어있는데, 
                        # 이렇게 폴더 구조로 구분했을 경우 이게 라벨이 된다. 
)
# Found 120 images belonging to 2 classes. 테스트는 120장

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022A82218550>
print(xy_train[0]) # 라벨값이 나온다. y는 batch size.
print(xy_train[0][0]) # x값
print(xy_train[0][1]) # y값
# print(xy_train[0][2]) # 없음

print(xy_train[0][0].shape, xy_train[0][1].shape) # (5, 150, 150, 3) (5,) => (batch size, target size, color) (batch size)
print(xy_test[0][0].shape, xy_test[0][1].shape) # (5, 150, 150, 3) (5,) => (batch size, target size, color) (batch size)

np.save('./_save/_npy/k59_3_x_train.npy', arr=xy_train[0][0])
np.save('./_save/_npy/k59_3_y_train.npy', arr=xy_train[0][1])
np.save('./_save/_npy/k59_3_x_test.npy', arr=xy_test[0][0])
np.save('./_save/_npy/k59_3_y_test.npy', arr=xy_test[0][1])
# print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

