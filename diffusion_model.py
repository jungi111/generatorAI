import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 모델 정의
class UNet(nn.Module):
    def __init__(self, channels, embed_dim=512):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + embed_dim, 64, 3, 2, 1),  # 첫 번째 합성곱 층의 채널 수를 64로 설정
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32 -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x64 -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),  # 128x128 -> 256x256
        )

    def forward(self, x, t):
        t = t.float()
        t = self.time_embedding(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        t = t.expand(x.size(0), t.size(1), x.size(2), x.size(3))
        x = torch.cat((x, t), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# DDPM 스케줄러
def linear_beta_schedule(timesteps, beta1, beta2):
    return torch.linspace(beta1, beta2, timesteps)

# DDPM 손실 함수
def ddpm_loss(model, x_0, t, beta_t):
    noise = torch.randn_like(x_0)  # 랜덤 노이즈 생성
    x_t = x_0 * torch.sqrt(1 - beta_t) + noise * torch.sqrt(beta_t)  # 노이즈 추가
    noise_pred = model(x_t, t)
    return nn.functional.l1_loss(noise_pred, noise)  # MAE 사용

# 학습 함수
def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    timesteps,
    model_save_path,
):
    model.train()

    plt.ion()  # 인터랙티브 모드 활성화
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title("Generated Image")
    ax.axis("off")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (x, _) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device).long()
            beta_t = scheduler[t].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            loss = ddpm_loss(model, x, t, beta_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if (i + 1) % 150 == 0:  # 배치 인덱스를 기준으로 50 배치마다 시각화
            #     model.eval()
            #     with torch.no_grad():
            #         sample_img = sample(
            #             model, x.size(2), x.size(1), timesteps, device, scheduler
            #         )
            #     sample_img = (
            #         sample_img.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze()
            #     )
            #     sample_img = (sample_img + 1) / 2  # Normalize to [0, 1]
            #     sample_img = np.clip(sample_img, 0, 1)  # 값을 [0, 1] 범위로 클리핑

            #     ax.clear()
            #     ax.imshow(sample_img)
            #     ax.set_title("Generated Image")
            #     ax.axis("off")
            #     plt.pause(0.01)
            #     model.train()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

        checkpoint_path = f"{model_save_path}_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plt.ioff()  # 인터랙티브 모드 비활성화
    plt.show()  # 시각화 창 표시

# 이미지 생성 함수
def sample(model, image_size, channels, timesteps, device, scheduler):
    model.eval()
    with torch.no_grad():
        x = torch.randn(
            1, channels, image_size, image_size, device=device
        )
        for t in range(timesteps - 1, -1, -1):
            t_tensor = torch.full((1,), t, device=device).long()
            beta_t = (
                scheduler[t]
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(1, channels, image_size, image_size)
            )
            noise_pred = model(x, t_tensor)
            x = x - beta_t * noise_pred
            x = x / torch.sqrt(1 - beta_t)
        x = torch.clamp(x, 0, 1)
        return x

def main():
    # 하이퍼파라미터 설정
    image_size = 32
    channels = 3
    batch_size = 128
    num_epochs = 100
    lr = 0.0001
    beta1 = 0.0001
    beta2 = 0.02
    timesteps = 2000
    model_save_path = "./ddpm_unet.pth"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./cifar10/train", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=6
    )

    model = UNet(channels, embed_dim=512).to(device)  # 임베딩 차원을 256으로 설정
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = linear_beta_schedule(timesteps, beta1, beta2).to(device)
    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        num_epochs,
        device,
        timesteps,
        model_save_path,
    )

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    print(f"Model loaded from {model_save_path}")

    samples = sample(model, image_size, channels, timesteps, device, scheduler)
    samples = samples.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze()
    samples = (samples + 1) / 2  # Normalize to [0, 1]

    plt.figure(figsize=(8, 8))
    plt.imshow(samples)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
