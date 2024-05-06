import os
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Input,
    Concatenate,
    Reshape,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    GlobalAveragePooling1D,
    Flatten,
    Add,
    RepeatVector,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 데이터 불러오기
current_directory = os.getcwd()
relative_path = os.path.join("dataset", "sentences.csv")
csv = pd.read_csv(os.path.join(current_directory, relative_path), nrows=50000)

# 결측값 처리
csv = csv.dropna()  # 결측값이 있는 행 삭제

# 데이터 처리
sentences = csv["TEXT_SENTENCE"]
level = csv["CURRICULUM_STEP_NO"]

# 데이터 분할
train_sentences, val_sentences, train_levels, val_levels = train_test_split(
    sentences, level, test_size=0.2, random_state=42, stratify=level
)

# 난이도 토크나이징 및 시퀀스화
level_tokenizer = Tokenizer()
level_tokenizer.fit_on_texts(train_levels.astype(str))

# 문장 토크나이징 및 시퀀스화
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(train_sentences)
total_words = len(tokenizer.word_index) + 1

# <start> 및 <end> 토큰 추가
special_tokens = ["<start>", "<end>"]
for token in special_tokens:
    if token not in tokenizer.word_index:
        tokenizer.word_index[token] = len(tokenizer.word_index) + 1
        total_words += 1

# 시퀀스 생성 함수
def create_sequences(words, levels, tokenizer, level_tokenizer):
    input_sequences = []
    level_sequences = []
    for word, level in zip(words, levels):
        # 단어 시퀀스 생성
        token_list = tokenizer.texts_to_sequences([word])[0]
        n_gram_sequence = (
            [tokenizer.word_index["<start>"]]
            + token_list
            + [tokenizer.word_index["<end>"]]
        )

        # 난이도 시퀀스 생성
        level_seq = level_tokenizer.texts_to_sequences([str(level)])[0]

        for i in range(1, len(n_gram_sequence)):
            input_sequences.append(n_gram_sequence[: i + 1])
            level_sequences.append(level_seq)

    return input_sequences, level_sequences

# 시퀀스 생성 (학습 및 검증 데이터)
train_input_sequences, train_level_sequences = create_sequences(
    train_sentences, train_levels, tokenizer, level_tokenizer
)
val_input_sequences, val_level_sequences = create_sequences(
    val_sentences, val_levels, tokenizer, level_tokenizer
)

# 패딩 추가
max_sequence_len = max([len(x) for x in train_input_sequences])
train_input_sequences = np.array(
    pad_sequences(train_input_sequences, maxlen=max_sequence_len, padding="pre")
)
val_input_sequences = np.array(
    pad_sequences(val_input_sequences, maxlen=max_sequence_len, padding="pre")
)

# 특징과 레이블 분리
X_train = train_input_sequences[:, :-1]
y_train = train_input_sequences[:, -1]
X_val = val_input_sequences[:, :-1]
y_val = val_input_sequences[:, -1]

# 난이도 정보를 특징에 포함
X_level_train = np.array(train_level_sequences)
X_level_val = np.array(val_level_sequences)

# Transformer 블록 정의
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(inputs, inputs)
    attention_output = Dense(inputs.shape[-1])(
        attention_output
    )  # 차원을 inputs의 마지막 차원 크기로 조정
    attention_output = Dropout(dropout)(attention_output)
    attention_output = Add()([attention_output, inputs])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    feedforward = Dense(ff_dim, activation="relu")(attention_output)
    feedforward = Dropout(dropout)(feedforward)
    feedforward = Dense(inputs.shape[-1])(
        feedforward
    )  # 차원을 inputs의 마지막 차원 크기로 유지
    feedforward = Dropout(dropout)(feedforward)
    output = Add()([feedforward, attention_output])
    return LayerNormalization(epsilon=1e-6)(output)

# 모델 구성
def build_model():
    word_input = Input(shape=(max_sequence_len - 1,), dtype="int32", name="word_input")
    level_input = Input(shape=(1,), dtype="int32", name="level_input")

    word_embedding = Embedding(input_dim=total_words, output_dim=128)(word_input)
    level_embedding = Embedding(input_dim=np.max(X_level_train) + 1, output_dim=128)(
        level_input
    )
    level_embedding = Reshape((128,))(level_embedding)
    level_embedding_repeated = RepeatVector(max_sequence_len - 1)(level_embedding)

    # 레이어 연결
    concat_layer = Concatenate(axis=-1)([word_embedding, level_embedding_repeated])

    # 다중 Transformer 블록 적용
    transformer_output = concat_layer
    for _ in range(2):  # 블록 수를 2개로 줄임
        transformer_output = transformer_block(
            transformer_output, head_size=64, num_heads=4, ff_dim=128, dropout=0.1
        )

    # 최종 출력 레이어
    transformer_output = GlobalAveragePooling1D()(transformer_output)
    flattened = Flatten()(transformer_output)
    flattened = BatchNormalization()(flattened)
    flattened = Dropout(0.2)(flattened)

    output_layer = Dense(total_words, activation="softmax")(flattened)

    model = Model(inputs=[word_input, level_input], outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # 학습률 조정
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model

# 콜백 정의
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "sentence_generation_model.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1
)

# 학습 여부에 따라 모델을 학습하거나 불러와서 사용
train_model = True

if train_model:
    model = build_model()

    # 모델 학습
    history = model.fit(
        [X_train, X_level_train],
        y_train,
        epochs=50,
        validation_data=([X_val, X_level_val], y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1,
    )

    # 최적의 에포크 찾기
    optimal_epoch = np.argmin(history.history["val_loss"]) + 1
    # 모델 학습
    model.fit([X_train, X_level_train], y_train, epochs=optimal_epoch, verbose=1)
    # 학습된 모델 저장
    tf.keras.models.save_model(model, "transformer_sentence_model_50000.keras")

else:
    # 저장된 모델 불러오기
    model = tf.keras.models.load_model("transformer_sentence_model_50000.keras")

# 모든 난이도 확인
print("Level Tokenizer word index:", level_tokenizer.word_index)

# generate_sentence 함수 수정
def generate_sentence(step_no, n_sentences, temperature=1.0):
    # 난이도 정보 설정
    level_str = str(step_no)
    level_seq = level_tokenizer.texts_to_sequences([level_str])

    if not level_seq or not level_seq[0]:
        print(f"난이도 '{step_no}'에 해당하는 시퀀스를 찾을 수 없습니다.")
        return

    level_seq = np.array(level_seq).reshape(-1, 1)  # 입력 형태에 맞게 reshape

    # 문장 생성
    output_sentences = set()
    while len(output_sentences) < n_sentences:
        input_sequence = [tokenizer.word_index["<start>"]]  # 시작 토큰 추가
        generated_words = []

        for _ in range(max_sequence_len - 1):
            token_list = pad_sequences(
                [input_sequence], maxlen=max_sequence_len - 1, padding="pre"
            )

            predicted = model.predict([token_list, level_seq], verbose=0)[0]
            predicted = np.asarray(predicted).astype("float64")
            predicted = np.log(predicted) / temperature
            exp_preds = np.exp(predicted)
            predicted = exp_preds / np.sum(exp_preds)

            predicted_index = np.random.choice(range(len(predicted)), p=predicted)

            # 종료 토큰이거나 0이면 종료
            if predicted_index == tokenizer.word_index["<end>"] or predicted_index == 0:
                break

            input_sequence.append(predicted_index)
            output_word = tokenizer.index_word.get(predicted_index, "")
            generated_words.append(output_word)

        sentence = " ".join(generated_words)
        if sentence and sentence not in output_sentences:  # 빈 문장이 아니고 중복되지 않을 경우에만 추가
            output_sentences.add(sentence)

    for idx, sentence in enumerate(output_sentences):
        print(f"Generated sentence {idx + 1}: {sentence}")

    return list(output_sentences)

# 예시
generate_sentence("32", 5)