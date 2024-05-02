import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 불러오기
current_directory = os.getcwd()
relative_path = os.path.join("dataset", "sentences.csv")
csv = pd.read_csv(os.path.join(current_directory, relative_path), nrows=60000)

# 결측값 처리
csv = csv.dropna()  # 결측값이 있는 행 삭제

# 데이터 처리
sentences = csv["TEXT_SENTENCE"]  # 문장 데이터
level = csv["CURRICULUM_STEP_NO"]  # 난이도 데이터

# Initialize level_to_sentences_indices
level_to_sentences_indices = {}

# Populate level_to_sentences_indices
for index, lvl in enumerate(level):
    if lvl not in level_to_sentences_indices:
        level_to_sentences_indices[lvl] = [index]
    else:
        level_to_sentences_indices[lvl].append(index)

# 문장 토크나이징 및 시퀀스화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# 난이도 토크나이징 및 시퀀스화
level_tokenizer = Tokenizer()
level_tokenizer.fit_on_texts(level.astype(str))
level_seq = np.array(level_tokenizer.texts_to_sequences(level.astype(str)))

# 시퀀스 생성
input_sequences = []
level_sequences = []
for sentence_index, sentence in enumerate(sentences):
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)
        level_sequences.append(level_seq[sentence_index])

# 패딩 추가
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# 특징과 레이블 분리
X = input_sequences[:, :-1]
labels = input_sequences[:, -1]
y = to_categorical(labels, num_classes=total_words)

# 난이도 정보를 특징에 포함
X_level = np.array(level_sequences)  # 마지막 난이도 정보는 제외

# 모델 구성
sentence_input = Input(shape=(max_sequence_len - 1,), dtype="int32", name="sentence_input")
level_input = Input(shape=(1,), dtype="int32", name="level_input")

sentence_embedding = Embedding(input_dim=total_words, output_dim=64)(sentence_input)
level_embedding = Embedding(input_dim=np.max(level_seq) + 1, output_dim=64)(level_input)
level_embedding = Reshape((64,))(level_embedding)

# RepeatVector를 사용하여 level_embedding을 max_sequence_len-1만큼 반복
level_embedding_repeated = RepeatVector(max_sequence_len - 1)(level_embedding)

# 레이어 연결 수정
concat_layer = Concatenate(axis=-1)([sentence_embedding, level_embedding_repeated])
lstm_layer = LSTM(16)(concat_layer)
output_layer = Dense(total_words, activation="softmax")(lstm_layer)

model = Model(inputs=[sentence_input, level_input], outputs=output_layer)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 미니배치 크기 설정
batch_size = 8

# 학습 여부에 따라 모델을 학습하거나 불러와서 사용
train_model = True
find_epochs = True

if train_model:

    if find_epochs: 
        # 검증 데이터 준비 예시
        split_index = int(len(X) * 0.8)  # 전체 데이터의 80%를 학습에 사용하고 20%를 검증에 사용
        X_val = X[split_index:]
        X_level_val = X_level[split_index:]
        y_val = y[split_index:]

        # EarlyStopping 콜백 정의
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # 모델 학습
        history = model.fit(
            [X, X_level],
            y,
            epochs=200,
            batch_size=batch_size,  # 미니배치 크기 설정
            validation_data=([X_val, X_level_val], y_val),
            callbacks=[early_stopping],
            verbose=1,
        )

        # 최적의 에포크 찾기
        optimal_epoch = np.argmin(history.history["val_loss"]) + 1
    # 모델 학습
    model.fit(
        [X, X_level],
        y,
        epochs=optimal_epoch, # optimal_epoch
        batch_size=batch_size,  # 미니배치 크기 설정
        verbose=1,
    )
    # 학습된 모델 저장
    model.save("sentence_generation_model.keras")
else:
    # 저장된 모델 불러오기
    model = tf.keras.models.load_model("sentence_generation_model.keras")


# 문장 생성 함수
def generate_sentence(step_no, n_sentences):
    output_sentences = []
    level_str = str(step_no)  # 난이도를 문자열로 변환
    level_seq = level_tokenizer.texts_to_sequences([level_str])

    if not level_seq or not level_seq[0]:
        raise ValueError(f"난이도 '{step_no}'에 해당하는 시퀀스를 찾을 수 없습니다.")

    level_seq = np.array(level_seq).reshape(-1, 1)  # 입력 형태에 맞게 reshape

    available_indices = level_to_sentences_indices.get(step_no, [])
    if len(available_indices) == 0:
        print(f"난이도 '{step_no}'에 해당하는 문장이 없습니다. 대체 문장을 생성합니다.")
        return ["No sentence available for this difficulty level."] * n_sentences
    
    while len(output_sentences) < n_sentences:
        seed_sentence_index = random.choice(available_indices)
        seed_sentence = sentences[seed_sentence_index]

        output_sentence = seed_sentence
        while len(output_sentence.split()) < max_sequence_len - 1:
            token_list = tokenizer.texts_to_sequences([output_sentence])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

            predicted = model.predict([token_list, level_seq], verbose=0)
            predicted_index = np.random.choice(range(len(predicted[0])), p=predicted[0])
            output_word = tokenizer.index_word.get(predicted_index, "")
            if output_word:
                output_sentence += " " + output_word
            else:
                break
        
        output_sentences.append(output_sentence)

    return output_sentences

# 문장 생성 테스트
generated_sentences = generate_sentence(25, 1)
for sentence in generated_sentences:
    print(sentence)
