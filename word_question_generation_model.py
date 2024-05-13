import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFGPT2Model, GPT2Tokenizer, GPT2Config
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random

def load_dataset(data, tokenizer, max_length):
    questions = data['QUESTION'] + " " + data['MEANING']
    labels = data['ANSWER'] - 1

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

def create_model():
    config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    base_model = TFGPT2Model.from_pretrained('gpt2', config=config)
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    
    outputs = base_model(input_ids, attention_mask=attention_mask)[0][:, -1, :]
    classifier_output = tf.keras.layers.Dense(4, activation='softmax')(outputs)
    
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
    filtered_options = []
    for opt in options:
        if not any(opt != other and (opt in other or other in opt) for other in options):
            filtered_options.append(opt)
    return filtered_options

def main(isTrain=True):
    device = "/GPU:0" if tf.config.experimental.list_physical_devices('GPU') else "/CPU:0"
    print(f"Using device: {device}")

    current_directory = os.getcwd()
    relative_path = os.path.join("dataset", "words_question.csv")
    csv = pd.read_csv(os.path.join(current_directory, relative_path))
    train_data, val_data = train_test_split(csv, test_size=0.2, random_state=42)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss=SparseCategoricalCrossentropy(from_logits=False),  # Set from_logits to False
        metrics=['accuracy']
    )

    train_dataset = load_dataset(train_data, tokenizer, max_length=64).batch(32).shuffle(10000)
    val_dataset = load_dataset(val_data, tokenizer, max_length=64).batch(32)

    if isTrain:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)
        ]
        model.fit(train_dataset, validation_data=val_dataset, epochs=3, callbacks=callbacks)
        tf.keras.models.save_model(model, "best_model.keras")
    else:
        model = tf.keras.models.load_model("best_model.keras")

    difficulty_level = 33
    generated_question = generate_question(train_data, difficulty_level)
    print(generated_question)

if __name__ == '__main__':
    main(isTrain=True)  # Set True for training, False for using the trained model
