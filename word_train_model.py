import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
from collections import Counter

# 현재 디렉토리와 파일 경로 설정
current_directory = os.getcwd()
relative_path = os.path.join("dataset", "words.csv")
model_save_path = "saved_model"
# 모델 저장 경로 디렉토리 생성
os.makedirs(model_save_path, exist_ok=True)

# 데이터 불러오기
def load_data():
    csv = pd.read_csv(os.path.join(current_directory, relative_path))
    # 결측값 처리
    csv = csv.dropna()  # 결측값이 있는 행 삭제
    words = csv["QUESTION"].astype(str).tolist()  # 단어 데이터
    levels = csv["CURRICULUM_STEP_NO"].astype(str).tolist()  # 난이도 데이터
    return words, levels

# 데이터 처리
def tokenize_and_build_vocab(data):
    tokens = [word.split() for word in data]
    vocab = Counter(token for sublist in tokens for token in sublist)
    vocab = {word: idx+1 for idx, (word, _) in enumerate(vocab.most_common())}
    vocab['<pad>'] = 0
    return tokens, vocab

def encode_data(tokens, vocab):
    return [[vocab[token] for token in sentence] for sentence in tokens]

def preprocess_data(words, levels):
    word_tokens, word_vocab = tokenize_and_build_vocab(words)
    level_tokens, level_vocab = tokenize_and_build_vocab(levels)

    # Convert tokens to indices
    word_indices = encode_data(word_tokens, word_vocab)
    level_indices = encode_data(level_tokens, level_vocab)

    # Pad sequences
    max_len = max(len(seq) for seq in word_indices)
    word_indices = pad_sequence([torch.tensor(seq) for seq in word_indices], batch_first=True, padding_value=word_vocab['<pad>'])
    level_indices = pad_sequence([torch.tensor(seq) for seq in level_indices], batch_first=True, padding_value=0).squeeze()

    # Labels
    labels = word_indices[:, 1:].clone()
    word_indices = word_indices[:, :-1]

    return word_indices, level_indices, labels, len(word_vocab), max_len, word_vocab, level_vocab

# 모델 정의
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

# 데이터 로더 생성
def create_data_loaders(X, X_level, y, batch_size=32):
    dataset = TensorDataset(X, X_level, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 모델 학습
def train_model(model_parts, train_loader, val_loader, word_vocab, level_vocab, num_epochs=30, patience=2, device='cpu'):
    word_embedding, level_embedding, lstm, fc = model_parts
    word_embedding.to(device)
    level_embedding.to(device)
    lstm.to(device)
    fc.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(word_embedding.parameters()) + 
                           list(level_embedding.parameters()) + 
                           list(lstm.parameters()) + 
                           list(fc.parameters()), lr=0.001)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        word_embedding.train()
        level_embedding.train()
        lstm.train()
        fc.train()
        train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1} / {num_epochs} - Training") as pbar:
            for batch in pbar:
                X_batch, X_level_batch, y_batch = [x.to(device) for x in batch]
                optimizer.zero_grad()
                outputs = forward(word_embedding, level_embedding, lstm, fc, X_batch, X_level_batch)
                loss = criterion(outputs, y_batch[:, -1])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Accuracy calculation
                _, preds = torch.max(outputs, dim=1)
                correct_train_predictions += torch.sum(preds == y_batch[:, -1])
                total_train_samples += y_batch.size(0)

                avg_train_loss = train_loss / len(train_loader)
                train_accuracy = correct_train_predictions.float() / total_train_samples
                pbar.set_postfix({"Loss": avg_train_loss, "Accuracy": train_accuracy.item()})

        word_embedding.eval()
        level_embedding.eval()
        lstm.eval()
        fc.eval()
        val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0

        with tqdm(val_loader, desc=f"Epoch {epoch + 1} / {num_epochs} - Validation") as pbar:
            for batch in pbar:
                X_batch, X_level_batch, y_batch = [x.to(device) for x in batch]

                with torch.no_grad():
                    outputs = forward(word_embedding, level_embedding, lstm, fc, X_batch, X_level_batch)
                    loss = criterion(outputs, y_batch[:, -1])
                    val_loss += loss.item()
                    
                    # Accuracy calculation
                    _, preds = torch.max(outputs, dim=1)
                    correct_val_predictions += torch.sum(preds == y_batch[:, -1])
                    total_val_samples += y_batch.size(0)

                    avg_val_loss = val_loss / len(val_loader)
                    val_accuracy = correct_val_predictions.float() / total_val_samples

                    pbar.set_postfix({"Loss": avg_val_loss, "Accuracy": val_accuracy.item()})

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy.item():.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy.item():.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'word_embedding': word_embedding.state_dict(),
                'level_embedding': level_embedding.state_dict(),
                'lstm': lstm.state_dict(),
                'fc': fc.state_dict(),
                'word_vocab': word_vocab,
                'level_vocab': level_vocab
            }, os.path.join(model_save_path, "word_generation_model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered")
                break

    print("Training complete. Model saved.")

# Apple Silicon GPU 사용 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 전체 실행
def main(train_model_flag=True):
    words, levels = load_data()
    X, X_level, y, total_words, max_sequence_len, word_vocab, level_vocab = preprocess_data(words, levels)
    train_loader, val_loader = create_data_loaders(X, X_level, y)

    vocab_size = total_words
    level_vocab_size = len(level_vocab)

    model_parts = build_model(vocab_size, level_vocab_size)

    if train_model_flag:
        train_model(model_parts, train_loader, val_loader, word_vocab, level_vocab, device=device)

        print(f"Model and vocabularies saved to {model_save_path}")
    else:
        # 모델 및 토크나이저 로드
        if os.path.exists(os.path.join(model_save_path, "word_generation_model.pth")):
            checkpoint = torch.load(os.path.join(model_save_path, "word_generation_model.pth"))
            word_embedding, level_embedding, lstm, fc = build_model(vocab_size, level_vocab_size)
            word_embedding.load_state_dict(checkpoint['word_embedding'])
            level_embedding.load_state_dict(checkpoint['level_embedding'])
            lstm.load_state_dict(checkpoint['lstm'])
            fc.load_state_dict(checkpoint['fc'])
            word_vocab = checkpoint['word_vocab']
            level_vocab = checkpoint['level_vocab']
            print(f"Model and vocabularies loaded from {model_save_path}")
        else:
            print(f"Model path {model_save_path} does not exist.")
            return

if __name__ == "__main__":
    main()
