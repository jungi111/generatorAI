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
    data['QUESTION'] = data['QUESTION'].str.replace('[^\\w\\s]', '', regex=True).str.lower()
    data['MEANING'] = data['MEANING'].str.replace('[^\\w\\s]', '', regex=True).str.lower()
    for i in range(1, 5):
        data[f'DISTRACTOR_{i}'] = data[f'DISTRACTOR_{i}'].str.replace('[^\\w\\s]', '', regex=True).str.lower()
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
        nn.Linear(64, 4),
        nn.Softmax(dim=1)
    )
    return base_model, classifier

def forward_pass(base_model, classifier, input_ids, attention_mask):
    base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_state = base_outputs.last_hidden_state[:, -1, :]
    logits = classifier(hidden_state)
    return logits

def load_dataset(data, tokenizer, max_length):
    print("Loading dataset...")
    questions = data['QUESTION'] + " " + data['MEANING']
    labels = data['ANSWER'] - 1  # 라벨 인코딩

    input_ids = []
    attention_masks = []
    label_list = []

    for i in range(len(questions)):
        encodings = tokenizer(
            questions.iloc[i],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids.append(encodings['input_ids'].flatten())
        attention_masks.append(encodings['attention_mask'].flatten())
        label_list.append(torch.tensor(labels.iloc[i], dtype=torch.long))

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
        
            avg_loss = total_loss / (len(dataloader))
            accuracy = correct_predictions.float() / total_samples  # 변경된 부분

            pbar.set_postfix({"Loss": avg_loss, "Accuracy": accuracy.item()})  # 변경된 부분

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.float() / total_samples  # 변경된 부분
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

            avg_loss = total_loss / (len(dataloader))
            accuracy = correct_predictions.float() / total_samples  # 변경된 부분

            pbar.set_postfix({"Loss": avg_loss, "Accuracy": accuracy.item()})  # 변경된 부분

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.float() / total_samples  # 변경된 부분
    return avg_loss, accuracy

def main(isTrain=True):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()
    print("Optimizer, scheduler, and criterion set up")

    if isTrain:
        total_epoch = 30
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
