import os
import pandas as pd
import torch
import re
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    get_scheduler,
    AdamW,
)
import torch.nn as nn
from tqdm import tqdm
import random


def preprocess_data(data):
    print("Preprocessing data...")
    data["QUESTION"] = (
        data["QUESTION"].astype(str).str.replace(r"[^\w\s]", "", regex=True).str.lower()
    )
    data["ANSWER"] = (
        data["ANSWER"].astype(str).str.replace(r"[^\w\s]", "", regex=True).str.lower()
    )
    for i in range(1, 5):
        data[f"DISTRACTOR_{i}"] = (
            data[f"DISTRACTOR_{i}"]
            .astype(str)
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.lower()
        )
    return data


def encode_data(df, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in tqdm(df.iterrows(), desc="Encoding data"):
        text = f"{row['QUESTION']} [SEP] {row['MEANING']}, {row['DISTRACTOR_1']}, {row['DISTRACTOR_2']}, {row['DISTRACTOR_3']}, {row['DISTRACTOR_4']}"
        encoded_dict = tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"].squeeze(0))
        attention_masks.append(encoded_dict["attention_mask"].squeeze(0))
        labels.append(int(row["ANSWER"]) - 1)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def create_data_loaders(input_ids, attention_masks, labels, batch_size=16):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=3):
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader),
    )
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")

        model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0

        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1} - Training") as pbar:
            for batch in pbar:
                batch_inputs, batch_masks, batch_labels = batch
                batch_inputs, batch_masks, batch_labels = (
                    batch_inputs.to("cuda"),
                    batch_masks.to("cuda"),
                    batch_labels.to("cuda"),
                )

                model.zero_grad()
                outputs = model(
                    input_ids=batch_inputs,
                    attention_mask=batch_masks,
                    labels=batch_labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                correct_train_predictions += torch.sum(preds == batch_labels)
                total_train_samples += batch_labels.size(0)

                avg_train_loss = total_train_loss / len(train_dataloader)
                train_accuracy = correct_train_predictions.float() / total_train_samples

                pbar.set_postfix(
                    {"Loss": avg_train_loss, "Accuracy": train_accuracy.item()}
                )

        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy.item()}"
        )

        model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0

        with tqdm(val_dataloader, desc=f"Epoch {epoch + 1} - Validation") as pbar:
            for batch in pbar:
                batch_inputs, batch_masks, batch_labels = batch
                batch_inputs, batch_masks, batch_labels = (
                    batch_inputs.to("cuda"),
                    batch_masks.to("cuda"),
                    batch_labels.to("cuda"),
                )

                with torch.no_grad():
                    outputs = model(
                        input_ids=batch_inputs,
                        attention_mask=batch_masks,
                        labels=batch_labels,
                    )
                    val_loss = outputs.loss
                    total_val_loss += val_loss.item()
                    _, preds = torch.max(outputs.logits, dim=1)
                    correct_val_predictions += torch.sum(preds == batch_labels)
                    total_val_samples += batch_labels.size(0)

                    avg_val_loss = total_val_loss / len(val_dataloader)
                    val_accuracy = correct_val_predictions.float() / total_val_samples

                    pbar.set_postfix(
                        {"Loss": avg_val_loss, "Accuracy": val_accuracy.item()}
                    )

        print(
            f"Epoch {epoch+1}, Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy.item()}"
        )

def find_similar_words(seed_word, model, device, tokenizer, num_words=3):
    # 문장을 완성하여 비슷한 의미의 단어들을 생성
    input_ids = tokenizer.encode(seed_word + " is similar to", return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=num_words,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # 샘플링 활성화
        top_k=50  # 상위 50개 토큰 중 선택
    )

    similar_words = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        words = text.split()
        for word in words:
            if word.lower() != seed_word.lower() and word.isalpha() and word.lower() not in similar_words:
                similar_words.append(word.lower())
                if len(similar_words) >= num_words:
                    return similar_words
    return similar_words

def generate_question_and_options(
    generation_model,
    classification_model,
    tokenizer,
    data,
    difficulty_level,
    device,
    num_options=4,
):
    generation_model.eval()
    classification_model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # 난이도에 맞는 데이터를 필터링
    filtered_data = data[data["CURRICULUM_STEP_NO"] == difficulty_level]
    if filtered_data.empty:
        return "No data available for this difficulty level."

    # 난이도에 맞는 데이터 필터링은 제거
    # 단어 생성
    current_directory = os.getcwd()
    word_train_path = os.path.join(current_directory, "saved_model")
    model_path = os.path.join(word_train_path, "word_generation_model.pth")
    model_parts, word_vocab, level_vocab = load_model_and_vocab(model_path, device=device)
    
    max_sequence_len = 20  # 최대 시퀀스 길이
    generated_word = generate_word(model_parts, word_vocab, level_vocab, difficulty_level, max_sequence_len, device=device)
    
    print('Generated word:', generated_word)
    
    # 비슷한 단어 찾기
    similar_words = find_similar_words(generated_word, generation_model, device, tokenizer, num_options - 1)

    # 생성된 단어를 사용하여 질문 구성
    question = generated_word

    # 보기 생성
    if len(similar_words) < num_options - 1:
        print("Not enough similar words found, filling with random words.")
        while len(similar_words) < num_options - 1:
            similar_words.append(generated_word[::-1])  # 예시로 뒤집은 단어 추가

    options = similar_words + [generated_word]
    random.shuffle(options)
    answer_index = options.index(generated_word) + 1

    sentence = generate_sentence_with_word(
        generation_model, tokenizer, question, device, max_attempts=20
    )

    if sentence is None:
        return generate_question_and_options(
            generation_model,
            classification_model,
            tokenizer,
            data,
            difficulty_level,
            device
        )

    return {
        "question": question,
        "options": options[:num_options],
        "answer": answer_index,
        "sentence": sentence,
    }


def generate_sentence_with_word(
    model, tokenizer, word, device, max_length=50, max_attempts=20, attempt=0
):
    prompt = f"Create a sentence with the word '{word}':"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # 모델을 사용하여 문장 생성
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,  # 최대 길이 설정 (원하는 문장의 최대 길이에 맞게 조정)
        pad_token_id=tokenizer.eos_token_id,  # 문장 종료 토큰 사용
        num_return_sequences=1,  # 생성할 시퀀스의 수
        eos_token_id=tokenizer.eos_token_id,  # 종료 토큰 설정
        no_repeat_ngram_size=1,  # 반복을 피하기 위해 설정
        do_sample=True,  # 샘플링을 활성화
        top_k=40,  # 상위 40개 단어만 고려
        top_p=0.9,  # 누적 확률이 0.9 이하인 단어들만 고려
        temperature=0.7,  # 샘플링의 다양성을 조절
    )

    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 생성된 텍스트에서 프롬프트 부분 제거
    sentence = generated_text[len(prompt) :].strip()

    # 잘못된 문자가 포함된 경우 다시 시도
    sentence = re.sub(r"[\uFFFD]", "", sentence)

    # " " 사이의 문장만 추출
    match = re.search(r'"([^"]*)"', sentence)
    if match:
        sentence = match.group(1)

    # 특정 text 있으면 그 뒤 텍스트를 제외
    sentence = sentence.split(":")[0].strip()
    sentence = sentence.split(".")[0].strip()

    # 만약 생성된 문장에 특정 단어가 포함되지 않은 경우 다시 시도
    if word not in sentence:
        print(
            f"'{word}' not in generated sentence on attempt {attempt + 1}. Retrying..."
        )
        if attempt + 1 < max_attempts:
            return generate_sentence_with_word(
                model, tokenizer, word, device, max_length, max_attempts, attempt + 1
            )
        else:
            print("Max attempts reached. Generating new question and options.")
            return None

    return sentence

def generate_word(model_parts, word_vocab, level_vocab, difficulty_level, max_sequence_len, device='cpu'):
    word_embedding, level_embedding, lstm, fc = model_parts
    lstm.eval()

    level_input = torch.tensor([level_vocab[str(difficulty_level)]], device=device)
    level_embed = level_embedding(level_input)

    valid_start_words = [v for k, v in word_vocab.items() if k != '<pad>']
    current_input = torch.tensor([random.choice(valid_start_words)], device=device)
    
    generated_word = ''
    word_embed = word_embedding(current_input).unsqueeze(0)
    concat_embed = torch.cat((word_embed, level_embed.unsqueeze(0)), dim=-1)
        
    output, hidden = lstm(concat_embed)
    logits = fc(output.squeeze(0))
    probs = torch.softmax(logits, dim=-1)

    pad_index = word_vocab['<pad>']
    probs[0][pad_index] = 0  # Set the probability of the <pad> token to zero

    # Renormalize probabilities after setting <pad> to zero
    probs /= torch.sum(probs)

    next_token_id = torch.argmax(probs, dim=-1).item()
    next_word = {v: k for k, v in word_vocab.items()}.get(next_token_id, None)
        
    generated_word += next_word + ' '
    current_input = torch.tensor([next_token_id], device=device)

    return generated_word.strip()

def build_model(vocab_size, level_vocab_size, embedding_dim=64, lstm_units=20):
    word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    level_embedding = nn.Embedding(level_vocab_size, embedding_dim)
    lstm = nn.LSTM(embedding_dim * 2, lstm_units, batch_first=True)
    fc = nn.Linear(lstm_units, vocab_size)
    return word_embedding, level_embedding, lstm, fc

def forward(word_embedding, level_embedding, lstm, fc, word_input, level_input):
    word_embed = word_embedding(word_input)
    level_embed = level_embedding(level_input).unsqueeze(1).repeat(1, word_embed.size(1), 1)
    concat = torch.cat((word_embed, level_embed), dim=-1)
    lstm_out, _ = lstm(concat)
    out = fc(lstm_out[:, -1, :])
    return out

# 모델과 토크나이저 로드
def load_model_and_vocab(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    word_embedding, level_embedding, lstm, fc = build_model(len(checkpoint['word_vocab']), len(checkpoint['level_vocab']))
    word_embedding.load_state_dict(checkpoint['word_embedding'])
    level_embedding.load_state_dict(checkpoint['level_embedding'])
    lstm.load_state_dict(checkpoint['lstm'])
    fc.load_state_dict(checkpoint['fc'])
    
    word_vocab = checkpoint['word_vocab']
    level_vocab = checkpoint['level_vocab']

    word_embedding.to(device)
    level_embedding.to(device)
    lstm.to(device)
    fc.to(device)

    return (word_embedding, level_embedding, lstm, fc), word_vocab, level_vocab

def main(is_train=True):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    classification_model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", num_labels=4
    )
    classification_model.config.pad_token_id = tokenizer.pad_token_id
    classification_model.to(device)
    
     # 텍스트 생성 모델 로드
    generation_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    generation_model.to(device)
    
    current_directory = os.getcwd()
    relative_path = os.path.join(current_directory, "dataset", "words_question.csv")
    csv = pd.read_csv(relative_path)
    processed_data = preprocess_data(csv)
    
    model_save_path = os.path.join(current_directory, "trained_model")

    if is_train:
        optimizer = AdamW(classification_model.parameters(), lr=5e-5)
        input_ids, attention_masks, labels = encode_data(processed_data, tokenizer)
        print("Data encoding completed")

        train_dataloader, val_dataloader = create_data_loaders(
            input_ids, attention_masks, labels, batch_size=16
        )
        print("Data loaders created")

        train_model(classification_model, train_dataloader, val_dataloader, optimizer)

        # 모델 저장
        os.makedirs(model_save_path, exist_ok=True)
        classification_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        print(f"Classification model and tokenizer saved to {model_save_path}")
    else:
        # 모델 로드
        if os.path.exists(model_save_path):
            classification_model = GPT2ForSequenceClassification.from_pretrained(
                model_save_path
            )
            tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
            classification_model.to(device)
            print(f"Classification model and tokenizer loaded from {model_save_path}")
        else:
            print(f"Model path {model_save_path} does not exist.")
            return
    
    difficulty_level = 33
    
    generated_question = generate_question_and_options(
        generation_model,
        classification_model,
        tokenizer,
        processed_data,
        difficulty_level,
        device
    )
    print(f"Generated question: {generated_question}")


if __name__ == "__main__":
    main(is_train=False)
