import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from tqdm import tqdm

def preprocess_data(data):
    print("Preprocessing data...")
    data['QUESTION'] = data['QUESTION'].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.lower()
    data['ANSWER'] = data['ANSWER'].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.lower()
    for i in range(1, 5):
        data[f'DISTRACTOR_{i}'] = data[f'DISTRACTOR_{i}'].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.lower()
    return data

def create_model(dropout_rate=0.3):
    print("Creating model...")
    config = GPT2Config.from_pretrained('gpt2')
    base_model = GPT2Model.from_pretrained('gpt2', config=config)
    classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(config.hidden_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 4),  # 네 가지 선택지를 포함하는 분류 문제
    )
    return base_model, classifier

def forward_pass(base_model, classifier, input_ids, attention_mask):
    base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_state = base_outputs.last_hidden_state[:, -1, :]
    logits = classifier(hidden_state)
    return logits

def load_dataset(data, tokenizer, max_length):
    print("Loading dataset...")
    inputs = data['CURRICULUM_STEP_NO'].astype(str)
    answers = data['ANSWER']
    distractors = [data[f'DISTRACTOR_{i}'] for i in range(1, 5)]

    input_ids = []
    attention_masks = []
    label_list = []

    for i in range(len(inputs)):
        input_text = inputs.iloc[i]
        encodings = tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # 모든 distractor가 존재하는지 확인
        if pd.notna(answers.iloc[i]) and all(pd.notna(distractor.iloc[i]) for distractor in distractors):
            options = [answers.iloc[i]] + [distractor.iloc[i] for distractor in distractors]
            if len(options) != 5:
                continue  # 선택지 수가 맞지 않으면 건너뛰기
            
            random.shuffle(options)

            try:
                answer_index = options.index(answers.iloc[i])
                if answer_index >= 4:
                    continue
                input_ids.append(encodings['input_ids'].flatten())
                attention_masks.append(encodings['attention_mask'].flatten())
                label_list.append(torch.tensor(answer_index, dtype=torch.long))
            except ValueError as e:
                continue

    # 유효한 데이터가 있는지 확인
    if not (len(input_ids) == len(attention_masks) == len(label_list)):
        raise ValueError("Mismatch in dataset lengths. Please check your dataset.")

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(label_list)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset



def generate_question(data, level):
    print("Generating question...")
    filtered_data = data[data['CURRICULUM_STEP_NO'] == level]
    if filtered_data.empty:
        return "No data available for this difficulty level."

    selected_row = filtered_data.sample(1).iloc[0]
    question = selected_row['QUESTION']
    answer = selected_row['ANSWER']
    distractors = [
        selected_row['DISTRACTOR_1'],
        selected_row['DISTRACTOR_2'],
        selected_row['DISTRACTOR_3'],
        selected_row['DISTRACTOR_4']
    ]

    options = [answer] + distractors
    options = options[:4]  # 선택지가 4개가 되도록 보장
    random.shuffle(options)
    answer_index = options.index(answer)

    return {"question": question, "options": options, "answer": answer_index}

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
    return schedule

def train(base_model, classifier, dataloader, criterion, optimizer, device, epoch):
    base_model.train()
    classifier.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with tqdm(dataloader, desc=f"Epoch {epoch + 1} - Training") as pbar:
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            logits = forward_pass(base_model, classifier, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
        
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions.float() / total_samples

            pbar.set_postfix({"Loss": avg_loss, "Accuracy": accuracy.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.float() / total_samples
    return avg_loss, accuracy

def validate(base_model, classifier, dataloader, criterion, device, epoch):
    base_model.eval()
    classifier.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with tqdm(dataloader, desc=f"Epoch {epoch + 1} - Validation") as pbar:
        for batch in pbar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            logits = forward_pass(base_model, classifier, input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions.float() / total_samples

            pbar.set_postfix({"Loss": avg_loss, "Accuracy": accuracy.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.float() / total_samples
    return avg_loss, accuracy

def main(isTrain=True):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    current_directory = os.getcwd()
    relative_path = os.path.join("dataset", "words_question.csv")
    print(f"Loading data from {relative_path}")
    csv = pd.read_csv(os.path.join(current_directory, relative_path))
    
    # 데이터 전처리
    processed_data = preprocess_data(csv)
    print("Data preprocessing completed")

    train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    print("Data split into train and validation sets")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    train_dataset = load_dataset(train_data, tokenizer, max_length=64)
    val_dataset = load_dataset(val_data, tokenizer, max_length=64)
    print("Datasets loaded")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    print("Data loaders created")

    base_model, classifier = create_model(dropout_rate=0.4)
    base_model.to(device)
    classifier.to(device)
    print("Model created and moved to device")

    optimizer = AdamW(list(base_model.parameters()) + list(classifier.parameters()), lr=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)
    criterion = nn.CrossEntropyLoss()
    print("Optimizer, scheduler, and criterion set up")

    if isTrain:
        total_epoch = 5
        early_stopping_patience = 3
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(total_epoch):
            print(f"Epoch {epoch + 1} / {total_epoch}")
            train_loss, train_acc = train(base_model, classifier, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = validate(base_model, classifier, val_loader, criterion, device, epoch)

            print(f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'base_model_state_dict': base_model.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, 'best_model.pth')
                print("Best model saved")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping")
                    break
    else:
        print("Loading best model for evaluation")
        checkpoint = torch.load('best_model.pth')
        base_model.load_state_dict(checkpoint['base_model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    difficulty_level = 33
    generated_question = generate_question(train_data, difficulty_level)
    print(f"Generated question: {generated_question}")

if __name__ == '__main__':
    main(isTrain=True)
