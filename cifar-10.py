import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_cnn_block(in_channels, out_channels, kernel_size=3, pool_size=2):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
    )
    return block


def early_stopping(val_loss, best_score, counter, patience=2, min_delta=0.001):
    if best_score is None:
        best_score = -val_loss
        counter = 0
    elif -val_loss < best_score + min_delta:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            return True, best_score, counter
    else:
        best_score = -val_loss
        counter = 0
    return False, best_score, counter


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=20,
    early_stopping=None,
):
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.3
    )  # 학습률 스케줄러 추가

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):  # Number of epochs
        model.train()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        best_score = None
        counter = 0
        patience = 3  # Early stopping patience

        pbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels).item()
            total_samples += images.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            pbar.set_postfix({"loss": epoch_loss, "accuracy": epoch_acc})

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        # Validation phase
        model.eval()

        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                val_running_corrects += torch.sum(
                    torch.max(outputs, 1)[1] == labels
                ).item()
                val_total_samples += images.size(0)

            val_loss = val_running_loss / val_total_samples
            val_acc = val_running_corrects / val_total_samples

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        scheduler.step()

        stop, best_score, counter = early_stopping(
            val_loss, best_score, counter, patience
        )
        if stop:
            break

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history


def plot_training_history(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, "b", label="Training accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def main(is_train=True):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # CIFAR-10 학습 데이터셋과 데이터로더
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/train", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # CIFAR-10 테스트 데이터셋과 데이터로더
    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/test", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    # CIFAR-10 클래스 이름
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    print(f"Train Data shape: {len(trainset)}")
    print(f"Train Loader shape: {len(trainloader)}")
    print(f"Classes: {classes}")
    print(f"Test Data shape: {len(testset)}")
    print(f"Test Loader shape: {len(testloader)}")

    cnn_block1 = create_cnn_block(3, 64)
    cnn_block2 = create_cnn_block(64, 128)
    cnn_block3 = create_cnn_block(128, 256)

    fc = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(256 * 4 * 4, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 10),
    )

    if is_train:
        # 모델, 손실 함수, 옵티마이저 정의
        model = nn.Sequential(cnn_block1, cnn_block2, cnn_block3, nn.Flatten(), fc).to(
            device
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        train_loss, val_loss, train_acc, val_acc = train_model(
            model,
            trainloader,
            testloader,
            criterion,
            optimizer,
            device,
            early_stopping=early_stopping,
        )

        # 훈련 과정 시각화
        plot_training_history(train_loss, val_loss, train_acc, val_acc)

        torch.save(model.state_dict(), "cifar10_model.pth")  # 모델 저장
    else:
        model = nn.Sequential(cnn_block1, cnn_block2, cnn_block3, nn.Flatten(), fc).to(
            device
        )
        model.load_state_dict(torch.load("cifar10_model.pth"))
        model.to(device)


if __name__ == "__main__":
    main(is_train=True)
