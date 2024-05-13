import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFGPT2Model, GPT2Tokenizer, GPT2Config
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow_addons as tfa
import random
import keras_tuner as kt

def load_dataset(data, tokenizer, max_length):
    questions = data['QUESTION'] + " " + data['MEANING']
    labels = to_categorical(data['ANSWER'] - 1, num_classes=4)  # 원-핫 인코딩 적용

    dataset = tokenizer(
        questions.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": dataset['input_ids'], "attention_mask": dataset['attention_mask']},
        labels
    ))

    return dataset

def create_model(dropout_rate=0.3):
    config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    base_model = TFGPT2Model.from_pretrained('gpt2', config=config)
    input_ids = Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    
    outputs = base_model(input_ids, attention_mask=attention_mask)[0][:, -1, :]
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(128, activation='relu')(outputs)
    outputs = Dense(64, activation='relu')(outputs)
    classifier_output = Dense(4, activation='softmax')(outputs)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=classifier_output)
    return model

def generate_question(data, level):
    filtered_data = data[data['CURRICULUM_STEP_NO'] == level]
    if filtered_data.empty:
        return "No data available for this difficulty level."

    selected_row = filtered_data.sample(1).iloc[0]
    question = selected_row['QUESTION']
    meaning = selected_row['MEANING']
    distractors = [
        selected_row['DISTRACTOR_1'],
        selected_row['DISTRACTOR_2'],
        selected_row['DISTRACTOR_3'],
        selected_row['DISTRACTOR_4']
    ]

    options = list(set(distractors + [meaning]))
    options = [opt for opt in options if opt != meaning]
    clean_options = remove_substring_duplicates(options, meaning)

    if len(clean_options) >= 3:
        random.shuffle(clean_options)
        clean_options = clean_options[:3]
        clean_options.append(meaning)
        random.shuffle(clean_options)
        answer_index = clean_options.index(meaning) + 1
        return {"question": question, "options": clean_options, "answer": answer_index}
    return "No valid options available."

def remove_substring_duplicates(options, meaning):
    filtered_options = [opt for opt in options if opt != meaning and meaning not in opt]

    final_options = []
    for opt in filtered_options:
        if not any(opt != other and (opt in other or other in opt) for other in filtered_options):
            final_options.append(opt)

    return final_options

def step_decay_schedule(initial_lr=5e-4, decay_factor=0.9, step_size=1):
    def schedule(epoch):
        return initial_lr * (decay_factor ** (epoch // step_size))
    return LearningRateScheduler(schedule, verbose=1)

# 데이터 전처리 함수
def preprocess_data(data):
    # Data cleaning and preprocessing
    data['QUESTION'] = data['QUESTION'].str.replace('[^\w\s]', '', regex=True).str.lower()
    data['MEANING'] = data['MEANING'].str.replace('[^\w\s]', '', regex=True).str.lower()
    for i in range(1, 5):
        data[f'DISTRACTOR_{i}'] = data[f'DISTRACTOR_{i}'].str.replace('[^\w\s]', '', regex=True).str.lower()
    return data

def main(isTrain=True):
    device = "/GPU:0" if tf.config.experimental.list_physical_devices('GPU') else "/CPU:0"
    print(f"Using device: {device}")

    current_directory = os.getcwd()
    relative_path = os.path.join("dataset", "words_question.csv")
    csv = pd.read_csv(os.path.join(current_directory, relative_path))
    
    # 데이터 전처리
    processed_data = preprocess_data(csv)

    train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = create_model(dropout_rate=0.4)
    model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc'),
        tfa.metrics.F1Score(num_classes=4, average='macro', name='f1_score')  # F1 점수 추가
    ]
)

    train_dataset = load_dataset(train_data, tokenizer, max_length=64).batch(32).shuffle(10000)
    val_dataset = load_dataset(val_data, tokenizer, max_length=64).batch(32)

    if isTrain:
        lr_scheduler = step_decay_schedule(initial_lr=5e-5, decay_factor=0.9, step_size=2)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1),
            lr_scheduler
        ]
        model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=callbacks)
        tf.keras.models.save_model(model, "best_model.keras")
    else:
        model = tf.keras.models.load_model("best_model.keras", custom_objects={'TFGPT2Model': TFGPT2Model})

    difficulty_level = 33
    generated_question = generate_question(train_data, difficulty_level)
    print(generated_question)

if __name__ == '__main__':
    main(isTrain=True)  
