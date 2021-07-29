# 훈련데이터를 기존데이터 20% 더 할것
# 성과비교
# temp에 저장 후 결과확인 뒤 삭제
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math
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
x_train, x_test, y_train, y_test = train_test_split(xy[0][0], xy[0][1], train_size=0.7, shuffle=True, random_state=10)
end = time.time() - start

argument_size = math.ceil(x_train.shape[0] * 0.2)

randidx = np.random.randint(x_train.shape[0], size=argument_size)

x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()

x_argumented = x_argumented.reshape(x_argumented.shape[0], 300, 300, 3)
x_test = x_test.reshape(x_test.shape[0], 300, 300, 3)
x_train = x_train.reshape(x_train.shape[0], 300, 300, 3)

x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size),
                                    batch_size=argument_size, shuffle=False
                                    ).next()[0]

x_train = np.concatenate((x_train, x_argumented))
y_train = np.concatenate((y_train, y_argumented))

'''
# 
# '''


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
#증폭 후
# loss :  0.022860504686832428
# acc :  0.9947090148925781