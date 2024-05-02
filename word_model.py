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
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# 현재 디렉토리와 파일 경로 설정
current_directory = os.getcwd()
relative_path = os.path.join("dataset", "words.csv")

# 데이터 불러오기
csv = pd.read_csv(os.path.join(current_directory, relative_path))

# 데이터 처리
words = csv["QUESTION"]  # 단어 데이터
level = csv["CURRICULUM_STEP_NO"]  # 난이도 데이터

# 데이터 처리 (난이도 정보 포함)
# 난이도 토크나이징 및 시퀀스화
level_tokenizer = Tokenizer()
level_tokenizer.fit_on_texts(level.astype(str))
level_seq = np.array(level_tokenizer.texts_to_sequences(level.astype(str)))

# 단어 토크나이징 및 시퀀스화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)
total_words = len(tokenizer.word_index) + 1

# 시퀀스 생성
input_sequences = []
level_sequences = []
for word_index, word in enumerate(words):
    token_list = tokenizer.texts_to_sequences([word])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)
        level_sequences.append(level_seq[word_index])

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
word_input = Input(shape=(max_sequence_len - 1,), dtype="int32", name="word_input")
level_input = Input(shape=(1,), dtype="int32", name="level_input")

word_embedding = Embedding(input_dim=total_words, output_dim=64)(word_input)
level_embedding = Embedding(input_dim=np.max(level_seq) + 1, output_dim=64)(level_input)
level_embedding = Reshape((64,))(level_embedding)

# RepeatVector를 사용하여 level_embedding을 max_sequence_len-1만큼 반복
level_embedding_repeated = RepeatVector(max_sequence_len - 1)(level_embedding)

# 레이어 연결 수정
concat_layer = Concatenate(axis=-1)([word_embedding, level_embedding_repeated])
lstm_layer = LSTM(20)(concat_layer)
output_layer = Dense(total_words, activation="softmax")(lstm_layer)

model = Model(inputs=[word_input, level_input], outputs=output_layer)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 학습 여부에 따라 모델을 학습하거나 불러와서 사용
train_model = True

if train_model:

    # 검증 데이터 준비 예시
    split_index = int(
        len(X) * 0.8
    )  # 전체 데이터의 80%를 학습에 사용하고 20%를 검증에 사용
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
        epochs=5,
        validation_data=([X_val, X_level_val], y_val),
        callbacks=[early_stopping],
        verbose=1,
    )

    # 최적의 에포크 찾기
    optimal_epoch = np.argmin(history.history["val_loss"]) + 1
    # 모델 학습
    model.fit([X, X_level], y, epochs=optimal_epoch, verbose=1)
    # 학습된 모델 저장
    tf.keras.models.save_model(model, "word_generation_model.keras")
else:
    # 저장된 모델 불러오기
    model = tf.keras.models.load_model("word_generation_model.keras")


# 난이도에 해당하는 단어 목록 가져오기
def get_words_by_level(step_no):
    words_by_level = []
    for idx, lev in enumerate(level):
        if int(step_no) == lev:
            words_by_level.append(words[idx])
    if not words_by_level:
        print(f"No words found for level {step_no}")
    return words_by_level


# seed word를 해당 난이도의 단어 중 랜덤하게 선택
def select_seed_word_by_level(step_no):
    words_by_level = get_words_by_level(step_no)
    return random.choice(words_by_level)


# 단어 생성 함수
def generate_word(step_no, n_words):
    seed_word = select_seed_word_by_level(step_no)
    level_str = str(step_no)  # 난이도를 문자열로 변환
    level_seq = level_tokenizer.texts_to_sequences([level_str])

    if not level_seq or not level_seq[0]:
        raise ValueError(f"난이도 '{step_no}'에 해당하는 시퀀스를 찾을 수 없습니다.")

    level_seq = np.array(level_seq).reshape(-1, 1)  # 입력 형태에 맞게 reshape

    output_words = []  # 생성된 단어들을 저장할 리스트
    while len(output_words) < n_words:
        token_list = tokenizer.texts_to_sequences([seed_word])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )

        predicted = model.predict([token_list, level_seq], verbose=0)
        predicted_index = np.random.choice(range(len(predicted[0])), p=predicted[0])
        output_word = tokenizer.index_word.get(predicted_index, "")

        # 정규 표현식을 사용해 영어 단어인지 확인합니다.
        if (
            re.match("^[a-zA-Z]+$", output_word)
            and len(output_word) > 1
            and output_word not in output_words
        ):
            output_words.append(
                output_word
            )  # 중복되지 않고 영어인 단어만 리스트에 추가
        else:
            # 영어 단어가 아니거나 중복된 단어가 생성되었을 경우, 다음 단어 생성을 시도합니다.
            continue
    print("base:", output_words)
    return output_words  # 생성된 단어들을 리스트로 반환


model_name = "gpt2-medium"  # 사용할 GPT 모델의 이름 설정
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt_model = TFGPT2LMHeadModel.from_pretrained(model_name)


# 단어 생성 함수 수정
def generate_word_with_gpt(step_no, n_words, gpt_model, gpt_tokenizer):
    generated_words = set()  # 중복된 단어를 필터링하기 위한 집합
    generated_count = 0

    while generated_count < n_words:
        # seed word와 난이도를 토큰화
        seed_word = select_seed_word_by_level(step_no)
        input_text = seed_word + " " + str(step_no)  # seed word와 난이도 정보를 합침
        input_ids = gpt_tokenizer.encode(input_text, return_tensors="tf")

        # GPT 모델을 사용하여 단어 생성
        output = gpt_model.generate(
            input_ids,
            max_length=len(seed_word)
            + n_words * 2,  # seed word 길이와 생성할 단어의 최대 길이 설정
            pad_token_id=gpt_tokenizer.eos_token_id,  # 문장 종료 토큰 사용
            num_return_sequences=1,  # 생성할 시퀀스의 수
            eos_token_id=gpt_tokenizer.eos_token_id,  # 종료 토큰 설정
            early_stopping=False,  # early_stopping을 해제하여 경고를 제거
            num_beams=1,  # 빔 서치를 사용하여 생성, num_beams를 1로 설정
        )

        # 생성된 단어 디코딩
        decoded_output = gpt_tokenizer.decode(output[0], skip_special_tokens=True)

        # 생성된 단어 중에서 영어 단어만 추출하고 한 글자 단어는 제외
        generated_words.update(
            word
            for word in decoded_output.split()
            if re.match("^[a-zA-Z]{2,}$", word.lower())
        )
        generated_count = len(generated_words)

    # 필요한 수의 단어만 추출하여 반환
    return list(generated_words)[:n_words]


generate_word("33", 5)

generated_word = generate_word_with_gpt("33", 5, gpt_model, gpt_tokenizer)
print("gpt2:", generated_word)
