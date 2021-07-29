# 실습
# categorical_crossentropy와 sigmoid 조합
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model, load_model, Sequential
import time
import tensorflow as tf

'''
train_datagen = ImageDataGenerator(
    rescale=1./120,
)

start = time.time()
test_datagen = ImageDataGenerator(rescale=1./255)

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

np.save('D:/study/_save/_npy/keras59_8_x_train', arr=train[0][0])
np.save('D:/study/_save/_npy/keras59_8_y_train', arr=train[0][1])
np.save('D:/study/_save/_npy/keras59_8_x_test', arr=test[0][0])
np.save('D:/study/_save/_npy/keras59_8_y_test', arr=test[0][1])
'''
# 학습
# '''
x_train = np.load('./_save/_npy/keras59_8_x_train.npy')
# x_predict = np.load('./_save/_npy/k59_5_x_predict.npy')
y_train = np.load('./_save/_npy/keras59_8_y_train.npy')
x_test = np.load('./_save/_npy/keras59_8_x_test.npy')
y_test = np.load('./_save/_npy/keras59_8_y_test.npy')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(36, input_shape=(80, 80, 3), kernel_size=(5,5), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(2, activation='softmax'))

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