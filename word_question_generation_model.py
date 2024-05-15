import os
import pandas as pd
import torch
import re
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    get_scheduler,
    AdamW,
)
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


def generate_question_and_options(
    generation_model,
    classification_model,
    tokenizer,
    data,
    difficulty_level,
    num_options=4,
):
    generation_model.eval()
    classification_model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # 난이도에 맞는 데이터를 필터링
    filtered_data = data[data["CURRICULUM_STEP_NO"] == difficulty_level]
    if filtered_data.empty:
        return "No data available for this difficulty level."

    # 필터링된 데이터에서 랜덤으로 단어 선택
    selected_row = filtered_data.sample(1).iloc[0]
    word_to_generate_from = selected_row["QUESTION"]

    # 선택된 단어를 기반으로 유사한 단어 생성
    input_ids = tokenizer.encode(word_to_generate_from, return_tensors="pt").to("cuda")
    outputs = generation_model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        attention_mask=torch.ones_like(input_ids),
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_word = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 잘못된 문자 제거
    generated_word = re.sub(r"[^\w\s]", "", generated_word).replace("_", "")

    # 생성된 단어를 질문으로 사용
    question = generated_word

    # 모델을 사용하여 정답과 보기를 예측
    input_text = f"{question} [SEP] {selected_row['MEANING']}, {selected_row['DISTRACTOR_1']}, {selected_row['DISTRACTOR_2']}, {selected_row['DISTRACTOR_3']}, {selected_row['DISTRACTOR_4']}"
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to("cuda")
    outputs = classification_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=1).item()

    # 예측된 정답을 기반으로 보기를 구성
    answer = selected_row[f"DISTRACTOR_{predicted_label + 1}"]
    distractors = [
        selected_row[f"DISTRACTOR_{i}"]
        for i in range(1, 5)
        if i != (predicted_label + 1)
    ]

    options = distractors + [answer]
    random.shuffle(options)
    answer_index = options.index(answer) + 1

    sentence = generate_sentence_with_word(
        generation_model, tokenizer, question, max_attempts=20
    )

    if sentence is None:
        return generate_question_and_options(
            generation_model,
            classification_model,
            tokenizer,
            data,
            difficulty_level,
        )

    return {
        "question": question,
        "options": options[:num_options],
        "answer": answer_index,
        "sentence": sentence,
    }


def generate_sentence_with_word(
    model, tokenizer, word, max_length=50, max_attempts=20, attempt=0
):
    prompt = f"Create a sentence with the word '{word}':"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones_like(input_ids).to("cuda")

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
        top_k=40,  # 상위 50개 단어만 고려
        top_p=0.9,  # 누적 확률이 0.95 이하인 단어들만 고려
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
                model, tokenizer, word, max_length, max_attempts, attempt + 1
            )
        else:
            print("Max attempts reached. Generating new question and options.")
            return None

    return sentence


def main(is_train=True):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    current_directory = os.getcwd()

    relative_path = os.path.join(current_directory, "dataset", "words_question.csv")
    print(f"Loading data from {relative_path}")

    csv = pd.read_csv(relative_path)
    processed_data = preprocess_data(csv)
    print("Data preprocessing completed")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    classification_model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", num_labels=4
    )
    classification_model.config.pad_token_id = tokenizer.pad_token_id
    classification_model.to(device)
    optimizer = AdamW(classification_model.parameters(), lr=5e-5)

    model_save_path = os.path.join(current_directory, "trained_model")

    if is_train:
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

    # 텍스트 생성 모델 로드
    generation_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    generation_model.to(device)

    # 난이도 수준을 입력하여 질문 생성
    difficulty_level = 33
    generated_question = generate_question_and_options(
        generation_model,
        classification_model,
        tokenizer,
        processed_data,
        difficulty_level,
    )
    print(f"Generated question: {generated_question}")


if __name__ == "__main__":
    main(is_train=False)
