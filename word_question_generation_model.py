import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, get_linear_schedule_with_warmup
import numpy as np
import random
from tqdm import tqdm

# 훈련 및 검증 함수
def train_epoch(model, classifier, loader, optimizer, criterion, device, epoch, num_epochs, scheduler):
    model.train()
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=True)
    for batch in loop:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = classifier(outputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 학습률 스케줄러 업데이트

        # Accuracy 및 Loss 업데이트
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 에포크 중 현재 평균 손실 및 정확도 계산
        current_loss = running_loss / total
        current_accuracy = 100. * correct / total
        
        
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=current_loss, accuracy=current_accuracy)

    return running_loss / len(loader), 100. * correct / total

# 검증 함수
def validate_epoch(model, classifier, loader, criterion, device, min_val_loss, patience, patience_counter):
    model.eval()
    classifier.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader, leave=False)
        for batch in loop:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            logits = classifier(outputs)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_val_loss = running_loss / total
            loop.set_description(f"Validation Epoch")
            loop.set_postfix(val_loss=current_val_loss, val_accuracy=100. * correct/total)

        # Early Stopping 로직
        if current_val_loss < min_val_loss:
            min_val_loss = current_val_loss
            torch.save(model.state_dict(), 'best_model.pth')  # 최적의 모델 저장
            patience_counter = 0  # 리셋
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                return None  # Early Stopping 조건 충족

    return running_loss / len(loader), 100. * correct / total, min_val_loss, patience_counter



# 문제 생성 함수
def generate_question(data, level):
    filtered_data = data[data['CURRICULUM_STEP_NO'] == level]
    if filtered_data.empty:
        return "No data available for this difficulty level."

    while True:  # 올바른 옵션을 생성할 때까지 무한 루프
        selected_row = filtered_data.sample(1).iloc[0]
        question = selected_row['QUESTION']
        meaning = selected_row['MEANING']
        distractors = [
            selected_row['DISTRACTOR_1'],
            selected_row['DISTRACTOR_2'],
            selected_row['DISTRACTOR_3'],
            selected_row['DISTRACTOR_4']
        ]

        # 모든 옵션을 포함하되 중복을 허용하지 않도록 설정
        options = list(set(distractors + [meaning]))
        options = [opt for opt in options if opt != meaning]

        # 중복 단어를 포함하지 않는 옵션 선택
        clean_options = remove_substring_duplicates(options, meaning)

        # 옵션을 섞고, 정답 포함하여 4개 선택
        if len(clean_options) >= 3:
            random.shuffle(clean_options)
            clean_options = clean_options[:3]  # 첫 세 개 선택
            clean_options.append(meaning)  # 정답 추가
            random.shuffle(clean_options)  # 다시 섞기
            answer_index = clean_options.index(meaning) + 1  # 정답 위치 찾기
            return {"question": question, "options": clean_options, "answer": answer_index}

def remove_substring_duplicates(options, meaning):
    filtered_options = []
    # 옵션과 meaning이 중복되지 않도록 검사
    options = [opt for opt in options if opt != meaning and not (opt in meaning or meaning in opt)]
    for opt in options:
        if not any(opt != other and (opt in other or other in opt) for other in options):
            filtered_options.append(opt)
    return filtered_options

def main(isTrain=True):
    
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    # 데이터 준비
    current_directory = os.getcwd()
    relative_path = os.path.join("dataset", "words_question.csv")

    # 데이터 불러오기
    csv = pd.read_csv(os.path.join(current_directory, relative_path))

    train_data, val_data = train_test_split(csv, test_size=0.2, random_state=42)

    # 토크나이저 및 데이터셋 준비
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 데이터셋 준비
    def prepare_dataset(data, tokenizer, max_length):
        def tokenize_function(row):
            inputs = tokenizer.encode_plus(
                row['QUESTION'] + ' ' + row['MEANING'],
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'labels': torch.tensor(row['ANSWER'] - 1, dtype=torch.long)  # 정답 번호를 0부터 시작하도록 조정
            }
        return [tokenize_function(row) for index, row in data.iterrows()]

    train_dataset = prepare_dataset(train_data, tokenizer, max_length=64)
    val_dataset = prepare_dataset(val_data, tokenizer, max_length=64)

    # 데이터 로더 설정
    print("Train dataset length:", len(train_dataset))
    print("Validation dataset length:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 디바이스 설정 통일
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # GPT-2 모델 로드 및 설정
    config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2Model.from_pretrained('gpt2', config=config).to(device)
    classifier = nn.Linear(config.n_embd, 4).to(device)  # 4개 선택지
    
    num_epochs = 10  # 학습할 총 에포크 수
        
    # 옵티마이저 및 손실 함수 설정
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()
    
    # Early Stopping 관련 변수
    min_val_loss = np.inf
    patience_counter = 0
    patience = 7
    
    if isTrain:
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_epoch(model, classifier, train_loader, optimizer, criterion, device, epoch, num_epochs, scheduler)
            val_result = validate_epoch(model, classifier, val_loader, criterion, device, min_val_loss, patience, patience_counter)
            if val_result is None:  # Early Stopping 체크
                break
            else:
                val_loss, val_accuracy, min_val_loss, patience_counter = val_result
            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
            
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    else:
        model.load_state_dict(torch.load('model_weights.pth'))
        classifier.load_state_dict(torch.load('classifier_weights.pth'))
        model.eval()
        classifier.eval()

    difficulty_level = 33
    generated_question = generate_question(train_data, difficulty_level)
    print(generated_question)

if __name__ == '__main__':
    main()