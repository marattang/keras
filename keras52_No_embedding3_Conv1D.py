from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten
from tensorflow.python.keras.layers.convolutional import Conv1D

# 1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# 
x = token.texts_to_sequences(docs)
print(x)
# 크기가 일정하지 않는 문제 발생
pad_x = pad_sequences(x, padding='pre', maxlen=5)
print(pad_x)
print(pad_x.shape)

word_size = len(token.word_index)
print(word_size) # 27

print(np.unique(pad_x))
# 원핫 인코딩하면 (13, 5) -> (13, 5, 27)
# 옥스포드 (13, 5, 100000) -> 6500만개
# embedding이 벡터화해주니까 원핫인코딩 할 필요 x, 하기 전의 데이터를 가지고 embedding을 하게 된다.
# 물론 크기는 맞춰줘야 함. 기본적으로 input layer에서 해준다.

# 2. 모델

pad_x = pad_x.reshape(13,5,1)
model = Sequential() # 인풋은 (13, 5)
# model.add(Embedding(input_dim=28, output_dim=77, input_length=5)) # param = input_dim * output_dim
# input_dim = 라벨의 수, 단어 사전의 갯수, input_length = 한 문장의 단어수(최대 길이), output_dim = RNN에서 filter 등.
# model.add(Embedding(input_dim=27, output_dim=77)) = model.add(Embedding(27, 77))
model.add(Conv1D(32, kernel_size=1, input_shape=(5,1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# embedding에서는 length가 먹히지 않음. 그냥 벡터연산을 하기 때문에 label의 갯수가 중요함. 연산에 영향을 미치는건 output_dim과 input_dim
# 두가지 length는 연산에 영향을 미치지 않음.
'''
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 5, 32)             64
_________________________________________________________________
flatten (Flatten)            (None, 160)               0
_________________________________________________________________
dense (Dense)                (None, 16)                2576
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17
=================================================================
Total params: 2,657
Trainable params: 2,657
Non-trainable params: 0
_________________________________________________________________
acc :  0.9230769276618958
'''
# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)



