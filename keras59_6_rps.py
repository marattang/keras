import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# '''
train_datagen = ImageDataGenerator(
    rescale=1./255,
  )

start = time.time()
test_datagen = ImageDataGenerator(rescale=1./255)

xy = train_datagen.flow_from_directory(
    '../_data/rps',
    target_size=(300, 300),
    batch_size=2520,
    class_mode='sparse', # {'binary', None, 'categorical', 'input', 'sparse'}
    shuffle=True
)
end = time.time() - start

np.save('./_save/_npy/k59_6_x.npy', arr=xy[0][0])
np.save('./_save/_npy/k59_6_y.npy', arr=xy[0][1])
'''
# 
# '''
x = np.load('./_save/_npy/k59_6_x.npy')
y = np.load('./_save/_npy/k59_6_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=10)

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv2D(32, input_shape=(300, 300, 3), kernel_size=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

learning_start = time.time()

es = EarlyStopping(monitor='val_acc', mode='max', patience=5)
model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=16, callbacks=[es])
learning_end = (time.time() - learning_start)/60
print('학습 걸린 시간(분) : ', learning_end)

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])
# '''
'''
loss :  0.0016953140730038285,  val_loss: 9.6115e-04 - val_acc: 1.0000
acc :  1.0
'''