from keras_preprocessing.image import image_data_generator
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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

x_train = np.concatenate((x_argumented, x_train))
y_train = np.concatenate((y_argumented, y_train))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

