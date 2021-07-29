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

# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
#     '../_data/brain/train',
#     target_size=(150, 150),
#     batch_size=5,
#     class_mode='binary', 
#     shuffle=True
# )

# 1. ImageDataGenerator 정의
# 2. 파일에서 땡겨올려면 -> flow_from_directory() // x, y가 튜플 행태로 뭉쳐있다.
# 3. 데이터에서 땡겨올려면 -> flow()              // x, y가 나뉘어 있다.
argument_size = 49
x_data = train_datagen.flow(
        np.tile(x_train[0].reshape(28*28), argument_size).reshape(-1, 28, 28, 1), # x_train하나만 들어가면 되는데 100장으로 늘리기 위한 소스
        np.zeros(argument_size), # zero는 shape모양을 맞춰주기 위해서 넣는다.
        batch_size=argument_size,
        shuffle=False
).next()
# 데이터를 반환할 때 iterator방식으로 반환한다. -> batch size만큼 반환을 하는데, 뭔가 순서가 있다.

# 바뀐 모양을 보기 위해서 tile로 100개 반복해서 보여주고, y값은 임의로 0을 넣는다.?

print(type(x_data)) # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
                    # -> <class 'tuple'> (next)
print(type(x_data[0])) # <class 'tuple'> 
                        # -> # <class 'numpy.ndarray'> (next)
print(x_data[0][0].shape)
                       # (28, 28, 1)    
print(x_data[0].shape) # (100, 28, 28, 1) -> x값 
                          # (28, 28, 1)
print(x_data[1].shape) # (100,) y 값
                          #  next 사용시 x_data[1]에서 y값이 나옴. (100,)

# next를 사용하면 전체가 다 실행된다. 전체를 실행시키기 위해서는 .next를 붙여서 실행시켜주면 된다. list에서 for문 쓴게 iterator.
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()