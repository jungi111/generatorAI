import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.init as init


# 모델 정의
class UNet(nn.Module):
    def __init__(self, channels, embed_dim=512):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + embed_dim, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),  # BatchNorm2d -> GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),  # BatchNorm2d -> GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256),  # BatchNorm2d -> GroupNorm
            nn.ReLU(inplace=True),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),  # BatchNorm2d -> GroupNorm
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),  # BatchNorm2d -> GroupNorm
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

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
def ddpm_loss(model, x_0, t, alpha_bar):
    noise = torch.randn_like(x_0)  # 랜덤 노이즈 생성
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise  # 노이즈 추가
    noise_pred = model(x_t, t)
    return nn.MSELoss()(noise_pred, noise)  # MSE 손실 함수 사용


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
            t = torch.randint(1, timesteps + 1, (x.size(0),), device=device).long()
            alpha_bar_t = scheduler[t].cumprod(dim=0)  # \bar{\alpha}_t 계산
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1).to(x.device)  # 브로드캐스팅을 위해 view 사용
            loss = ddpm_loss(model, x, t, alpha_bar_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (i + 1) % 150 == 0:
                model.eval()
                with torch.no_grad():
                    sample_img = sample(
                        model, x.size(2), x.size(1), timesteps, device, scheduler
                    )
                sample_img = (
                    sample_img.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze()
                )
                sample_img = (sample_img + 1) / 2  # Normalize to [0, 1]
                sample_img = np.nan_to_num(sample_img) 
                sample_img = np.clip(sample_img, 0, 1)  # 값을 [0, 1] 범위로 클리핑

                ax.clear()
                ax.imshow(sample_img)
                ax.set_title("Generated Image")
                ax.axis("off")
                plt.pause(0.01)
                model.train()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

        checkpoint_path = f"{model_save_path}_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print("Training Finish")

    plt.ioff()  # 인터랙티브 모드 비활성화
    plt.show()  # 시각화 창 표시


# 이미지 생성 함수
def sample(model, image_size, channels, timesteps, device, scheduler):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, channels, image_size, image_size, device=device)  # x_T 샘플링
        for t in range(timesteps, 0, -1):
            t_index = t - 1  # 0부터 시작하는 인덱스로 조정
            t_tensor = torch.full((1,), t, device=device).long()
            alpha_t = scheduler[t_index]
            alpha_t = alpha_t.view(-1, 1, 1, 1).to(x.device)
            sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - scheduler[:t_index+1].cumprod(dim=0))[-1]

            # t > 1 인 경우 z를 샘플링하고 그렇지 않으면 z = 0
            if t > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            beta_t = scheduler[t_index]
            sigma_t = torch.sqrt(beta_t)
            epsilon_theta = model(x, t_tensor)
            x = sqrt_recip_alpha_t * (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * epsilon_theta) + sigma_t * z
        
        x = torch.clamp(x, -1, 1)  # Normalize to [-1, 1]
        return x
    
def add_noise(x_0, timesteps, scheduler):
    x_t = x_0
    for t in range(timesteps, 0, -1):
        noise = torch.randn_like(x_0)
        alpha_t = scheduler[t-1]
        x_t = torch.sqrt(alpha_t) * x_t + torch.sqrt(1 - alpha_t) * noise
    return x_t

def validate(model, x_t, timesteps, scheduler, device):
    model.eval()
    with torch.no_grad():
        for t in range(timesteps, 0, -1):
            t_index = t - 1
            t_tensor = torch.full((1,), t, device=device).long()
            alpha_t = scheduler[t_index]
            alpha_t = alpha_t.view(-1, 1, 1, 1).to(x_t.device)
            sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - scheduler[:t_index+1].cumprod(dim=0))[-1]
            if t > 1:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
            beta_t = scheduler[t_index]
            sigma_t = torch.sqrt(beta_t)
            epsilon_theta = model(x_t, t_tensor)
            x_t = sqrt_recip_alpha_t * (x_t - (1 - alpha_t) / sqrt_one_minus_alpha_bar * epsilon_theta) + sigma_t * z
        x_t = torch.clamp(x_t, -1, 1)
        return x_t


def main():
    # 하이퍼파라미터 설정
    image_size = 32
    channels = 3
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.0001
    beta1 = 0.0001
    beta2 = 0.02
    timesteps = 1000
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
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./cifar10/train", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=6
    )

    model = UNet(channels, embed_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
