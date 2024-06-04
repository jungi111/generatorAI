import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 환경변수 설정
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 모델 정의
class UNet(nn.Module):
    def __init__(self, channels, embed_dim=512):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + embed_dim, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
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
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        t = self.time_embedding(t.float().unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        t = t.expand(x.size(0), t.size(1), x.size(2), x.size(3))
        x = torch.cat((x, t), dim=1)
        x = self.encoder(x)
        if torch.isnan(x).any():
            print("NaN detected in encoder output.")
        x = self.decoder(x)
        if torch.isnan(x).any():
            print("NaN detected in decoder output.")
        return x


# DDPM 스케줄러
def linear_beta_schedule(timesteps, beta1, beta2):
    betas = torch.linspace(beta1, beta2, timesteps)
    betas = torch.clamp(betas, min=1e-4)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return torch.clamp(alpha_bars, min=1e-4)


# DDPM 손실 함수
def ddpm_loss(model, x_0, t, alpha_bar_t):
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    noise_pred = model(x_t, t)
    if torch.isnan(x_t).any():
        print("NaN detected in x_t during loss calculation.")
    if torch.isnan(noise_pred).any():
        print("NaN detected in noise_pred during loss calculation.")
    return nn.MSELoss()(noise_pred, noise)


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

    plt.ion()
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
            alpha_bar_t = scheduler[t].view(-1, 1, 1, 1).to(x.device)
            loss = ddpm_loss(model, x, t, alpha_bar_t)
            optimizer.zero_grad()
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                    + 1
                ) / 2
                sample_img = np.clip(np.nan_to_num(sample_img), 0, 1)

                ax.clear()
                ax.imshow(sample_img)
                ax.set_title("Generated Image")
                ax.axis("off")
                plt.pause(0.01)
                model.train()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
        torch.save(model.state_dict(), f"{model_save_path}_checkpoint.pth")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    plt.ioff()
    plt.show()


# 이미지 생성 함수
def sample(model, image_size, channels, timesteps, device, scheduler):
    model.eval()
    with torch.no_grad():
        xt = torch.randn((1, channels, image_size, image_size)).to(device)
        for t in range(timesteps, 0, -1):
            t_tensor = torch.tensor([t], device=xt.device).long()
            epsilon_theta = model(xt, t_tensor)
            alpha_t = scheduler[t - 1].clamp(min=1e-5).to(device)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            xt = (
                xt - (1 - alpha_t) / sqrt_one_minus_alpha_t * epsilon_theta
            ) / torch.sqrt(alpha_t)
            if t > 1:
                z = torch.randn_like(xt).to(device)
                xt += torch.sqrt(alpha_t) * z
            xt = torch.clamp(xt, -1.0, 1.0)
        return xt


# 메인 함수
def main():
    image_size = 32
    channels = 3
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.00001
    beta1 = 0.0001
    beta2 = 0.02
    timesteps = 1500
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
    samples = (samples.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze() + 1) / 2

    plt.figure(figsize=(8, 8))
    plt.imshow(samples)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
