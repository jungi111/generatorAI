import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_data(tickers):
    data_dict = {}
    for ticker in tickers:
        data_dict[ticker] = yf.download(ticker, start="2018-01-01", end="2024-04-30")["Close"]
    data = pd.DataFrame(data_dict)
    data = data.dropna()
    return data


# 데이터 정규화 및 텐서 변환 함수 정의
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    return data, scaler


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + tw : i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# LSTM 블록 정의 함수
def create_lstm_block(input_size=5, hidden_layer_size=200, num_layers=3, dropout=0.3, output_size=5):
    lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout)
    linear = nn.Linear(hidden_layer_size, output_size)
    return (
        lstm,
        linear,
        (torch.zeros(num_layers, 1, hidden_layer_size), torch.zeros(num_layers, 1, hidden_layer_size)),
    )



# 모델 학습 함수 정의
def train_model(
    lstm,
    linear,
    hidden_cell,
    train_sequences,
    train_labels,
    device,
    epochs,
    patience=10,
    min_delta=0.001,
    lr=0.0005,
):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(
        list(lstm.parameters()) + list(linear.parameters()), lr=lr
    )

    lstm.to(device)
    linear.to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    num_layers = hidden_cell[0].size(0)

    for epoch in range(epochs):
        lstm.train()
        linear.train()

        running_loss = 0.0
        total_samples = 0
        total_batches = len(train_sequences)

        pbar = tqdm(
            zip(train_sequences, train_labels),
            desc=f"Epoch {epoch+1}/{epochs}",
            unit="batch",
            total=total_batches,
        )
        for batch_idx, (seq, labels) in enumerate(pbar):
            seq, labels = seq.to(device), labels.to(device)

            optimizer.zero_grad()
            hidden_cell = (
                torch.zeros(num_layers, 1, hidden_cell[0].size(2)).to(device),
                torch.zeros(num_layers, 1, hidden_cell[0].size(2)).to(device),
            )

            lstm_out, hidden_cell = lstm(seq.view(len(seq), 1, -1), hidden_cell)
            y_pred = linear(lstm_out.view(len(seq), -1))[-1]

            # 예측값과 실제값의 크기 일치
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * seq.size(0)
            total_samples += seq.size(0)

            epoch_loss = running_loss / total_samples
            pbar.set_postfix(
                {"loss": epoch_loss, "batch": f"{batch_idx+1}/{total_batches}"}
            )

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Early Stopping Check
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping on epoch {epoch + 1}")
                break

    return lstm, linear, hidden_cell


def predict_future(lstm, linear, hidden_cell, data, scaler, fut_pred, train_window, device):
    lstm.eval()
    linear.eval()
    test_inputs = data[-train_window:].tolist()
    test_inputs = np.array(test_inputs)  # 리스트를 numpy 배열로 변환

    future_predictions = []
    num_layers = hidden_cell[0].size(0)

    for i in tqdm(range(fut_pred), desc="Predicting"):
        seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
        with torch.no_grad():
            hidden_cell = (
                torch.zeros(num_layers, 1, hidden_cell[0].size(2)).to(device),
                torch.zeros(num_layers, 1, hidden_cell[0].size(2)).to(device),
            )
            lstm_out, hidden_cell = lstm(seq.view(len(seq), 1, -1), hidden_cell)
            y_pred = linear(lstm_out.view(len(seq), -1))[-1]
            y_pred = y_pred.cpu().numpy()
            future_predictions.append(y_pred)
            test_inputs = np.append(test_inputs, [y_pred], axis=0)

    future_predictions = np.array(future_predictions)
    actual_predictions = scaler.inverse_transform(future_predictions.reshape(-1, data.shape[1]))
    return actual_predictions



def main(is_train=True):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    tickers = ["AAPL", "NVDA", "MSFT", "INTC", "KO"]

    # 데이터 확인
    data = load_data(tickers)

    train_window = 30
    processed_data, scaler = preprocess_data(data.values)

    sequences = create_inout_sequences(processed_data, train_window)

    # 텐서 변환
    train_sequences = [
        torch.FloatTensor(seq).reshape(train_window, -1) for seq, label in sequences
    ]
    train_labels = [torch.FloatTensor(label).reshape(-1) for seq, label in sequences]

    print(f"Total sequences: {len(train_sequences)}")

    lstm, linear, hidden_cell = create_lstm_block(input_size=len(tickers))

    if is_train:
        # LSTM 모델 학습 및 저장
        lstm, linear, hidden_cell = train_model(
            lstm,
            linear,
            hidden_cell,
            train_sequences,
            train_labels,
            device,
            epochs=100,
            patience=10,
            min_delta=0.001,
            lr=0.0005,
        )

        # 모델 저장
        torch.save(
            {
                "lstm_state_dict": lstm.state_dict(),
                "linear_state_dict": linear.state_dict(),
            },
            "combined_model.pth",
        )
    else:
        # 저장된 모델 로드
        checkpoint = torch.load("combined_model.pth")
        lstm.load_state_dict(checkpoint["lstm_state_dict"])
        linear.load_state_dict(checkpoint["linear_state_dict"])
        lstm.to(device)  # 모델을 올바른 장치로 이동
        linear.to(device)  # 모델을 올바른 장치로 이동

    # 미래 데이터 예측
    fut_pred = 960  # 2024년 5월부터 2026년 12월까지의 개월 수
    predictions = predict_future(
        lstm, linear, hidden_cell, data.values, scaler, fut_pred, train_window, device
    )

    # 예측된 데이터 크기 확인
    for i, ticker in enumerate(tickers):
        assert predictions.shape[0] == fut_pred, f"Prediction length for {ticker} is not 960"
        print(f"{ticker} predictions: {len(predictions[:, i])} (expected: 960)")

    # 예측 결과 시각화
    pred_dates = pd.date_range(start="2024-05-01", periods=fut_pred, freq="D")
    pred_series = pd.DataFrame(predictions, index=pred_dates, columns=tickers)

    plt.figure(figsize=(12, 8))
    plt.grid(True)

    for i, ticker in enumerate(tickers):
        combined_series = pd.Series(np.concatenate([data.values[:, i], pred_series[ticker]]), 
                                    index=np.concatenate([data.index, pred_series.index]))
        plt.plot(combined_series.index, combined_series.values, label=f"{ticker} Actual + Predicted")

    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xlim([pd.Timestamp('2018-01-01'), pd.Timestamp('2026-12-31')])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(is_train=False)
