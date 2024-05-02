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


class WordGenerator:
    def __init__(self):
        self.current_directory = os.getcwd()
        self.relative_path = os.path.join("dataset", "words.csv")
        self.csv = pd.read_csv(os.path.join(self.current_directory, self.relative_path))
        self.words = self.csv["QUESTION"]
        self.level = self.csv["CURRICULUM_STEP_NO"]

        self.level_tokenizer = Tokenizer()
        self.level_tokenizer.fit_on_texts(self.level.astype(str))
        self.level_seq = np.array(self.level_tokenizer.texts_to_sequences(self.level.astype(str)))

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.words)
        self.total_words = len(self.tokenizer.word_index) + 1

        self.max_sequence_len = self.prepare_sequences()

        self.model = self.build_model()
        self.train_model()

        self.model_name = "gpt2-medium"
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.gpt_model = TFGPT2LMHeadModel.from_pretrained(self.model_name)

    def prepare_sequences(self):
        input_sequences = []
        level_sequences = []
        for word_index, word in enumerate(self.words):
            token_list = self.tokenizer.texts_to_sequences([word])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[: i + 1]
                input_sequences.append(n_gram_sequence)
                level_sequences.append(self.level_seq[word_index])

        max_sequence_len = max([len(x) for x in input_sequences])
        self.input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))
        self.X = self.input_sequences[:, :-1]
        self.labels = self.input_sequences[:, -1]
        self.y = to_categorical(self.labels, num_classes=self.total_words)

        self.X_level = np.array(level_sequences)
        return max_sequence_len

    def build_model(self):
        word_input = Input(shape=(self.max_sequence_len - 1,), dtype="int32", name="word_input")
        level_input = Input(shape=(1,), dtype="int32", name="level_input")

        word_embedding = Embedding(input_dim=self.total_words, output_dim=64)(word_input)
        level_embedding = Embedding(input_dim=np.max(self.level_seq) + 1, output_dim=64)(level_input)
        level_embedding = Reshape((64,))(level_embedding)

        level_embedding_repeated = RepeatVector(self.max_sequence_len - 1)(level_embedding)

        concat_layer = Concatenate(axis=-1)([word_embedding, level_embedding_repeated])
        lstm_layer = LSTM(20)(concat_layer)
        output_layer = Dense(self.total_words, activation="softmax")(lstm_layer)

        model = Model(inputs=[word_input, level_input], outputs=output_layer)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train_model(self):
        # 학습 여부에 따라 모델을 학습하거나 불러와서 사용
        train_model = False

        if train_model:
            split_index = int(len(self.X) * 0.8)
            X_val = self.X[split_index:]
            X_level_val = self.X_level[split_index:]
            y_val = self.y[split_index:]

            early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

            history = self.model.fit(
                [self.X, self.X_level],
                self.y,
                epochs=300,
                validation_data=([X_val, X_level_val], y_val),
                callbacks=[early_stopping],
                verbose=1,
            )

            optimal_epoch = np.argmin(history.history["val_loss"]) + 1
            self.model.fit([self.X, self.X_level], self.y, epochs=optimal_epoch, verbose=1)
            self.model.save("word_generation_model.keras")
        else:
            # 저장된 모델 불러오기
            self.model.load_weights("word_generation_model.keras")


    def generate_word(self, step_no, n_words):
        seed_word = self.select_seed_word_by_level(step_no)
        level_str = str(step_no)
        level_seq = self.level_tokenizer.texts_to_sequences([level_str])

        if not level_seq or not level_seq[0]:
            raise ValueError(f"난이도 '{step_no}'에 해당하는 시퀀스를 찾을 수 없습니다.")

        level_seq = np.array(level_seq).reshape(-1, 1)

        output_words = []
        while len(output_words) < n_words:
            token_list = self.tokenizer.texts_to_sequences([seed_word])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len - 1, padding="pre")

            predicted = self.model.predict([token_list, level_seq], verbose=0)
            predicted_index = np.random.choice(range(len(predicted[0])), p=predicted[0])
            output_word = self.tokenizer.index_word.get(predicted_index, "")

            if (
                re.match("^[a-zA-Z]+$", output_word)
                and len(output_word) > 1
                and output_word not in output_words
            ):
                output_words.append(output_word)
            else:
                continue
        return output_words

    def generate_word_with_gpt(self, step_no, n_words):
        generated_words = set()
        generated_count = 0

        while generated_count < n_words:
            seed_word = self.select_seed_word_by_level(step_no)
            input_text = seed_word + " " + str(step_no)
            input_ids = self.gpt_tokenizer.encode(input_text, return_tensors="tf")

            output = self.gpt_model.generate(
                input_ids,
                max_length=len(seed_word) + n_words * 2,
                pad_token_id=self.gpt_tokenizer.eos_token_id,
                num_return_sequences=1,
                eos_token_id=self.gpt_tokenizer.eos_token_id,
                early_stopping=False,
                num_beams=1,
            )

            decoded_output = self.gpt_tokenizer.decode(output[0], skip_special_tokens=True)

            generated_words.update(
                word
                for word in decoded_output.split()
                if re.match("^[a-zA-Z]{2,}$", word.lower())
            )
            generated_count = len(generated_words)

        return list(generated_words)[:n_words]

    def get_words_by_level(self, step_no):
        words_by_level = []
        for idx, lev in enumerate(self.level):
            if int(step_no) == lev:
                words_by_level.append(self.words[idx])
        if not words_by_level:
            print(f"No words found for level {step_no}")
        return words_by_level

    def select_seed_word_by_level(self, step_no):
        words_by_level = self.get_words_by_level(step_no)
        return random.choice(words_by_level)
