import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset
import numpy as np
import os
import clip


# 두 개의 컨볼루션 레이어와 그룹 정규화(Group Normalization), GELU 활성화 함수를 포함한 이중 컨볼루션 블록. 잔차 연결(residual connection)을 옵션으로 포함
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


# 입력 피처 맵의 공간적 관계를 캡처하는 자가 주의(Self-Attention) 메커니즘
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


# 다운샘플링을 수행하며, 맥스 풀링, 이중 컨볼루션, 자가 주의 메커니즘을 포함. 또한 시간 임베딩 레이어를 포함하여 각 레이어에서 시간 정보를 추가
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512, device="cuda"):
        super().__init__()
        self.device = device
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            SelfAttention(in_channels),  # Add Self-Attention
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = (
            self.emb_layer(t)
            .to(self.device)[:, :, None, None]
            .repeat(1, 1, x.shape[-2], x.shape[-1])
        )
        return x + emb


# 업샘플링을 수행하며, 이중 컨볼루션, 자가 주의 메커니즘을 포함. 또한 시간 임베딩 레이어를 포함하여 각 레이어에서 시간 정보를 추가
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512, device="cuda"):
        super().__init__()
        self.device = device
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            SelfAttention(in_channels),  # Add Self-Attention
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = (
            self.emb_layer(t)
            .to(self.device)[:, :, None, None]
            .repeat(1, 1, x.shape[-2], x.shape[-1])
        )
        output = x + emb
        return output


class UNet_classifier(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim, device=device)
        self.down2 = Down(128, 256, emb_dim=time_dim, device=device)
        self.down3 = Down(256, 256, emb_dim=time_dim, device=device)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 256, emb_dim=time_dim, device=device)
        self.up2 = Up(384, 128, emb_dim=time_dim, device=device)
        self.up3 = Up(192, 64, emb_dim=time_dim, device=device)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        t = t.unsqueeze(-1).to(self.device)  # [batch_size, 1]
        pos_enc_a = torch.sin(t * inv_freq)
        pos_enc_b = torch.cos(t * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output


# 디퓨전 프로세스를 관리하는 클래스. 노이즈 스케줄링, 이미지 노이즈 추가 및 제거, 샘플링 등의 기능을 포함
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
    ):
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).to(self.device)[
            :, None, None, None
        ]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).to(self.device)[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def forward(self, x, t):
        noise = torch.randn_like(x).to(self.device)
        alphas_cumprod_t = self.alpha_hat[t].view(-1, 1, 1, 1).to(self.device)
        sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
        return sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise

    def sample_with_classifier(self, model, n, target_class=None):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            if target_class is not None:
                y = torch.tensor([target_class] * n).to(self.device)
            else:
                y = None
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, y)
                alpha = self.alpha[t][:, None, None, None].to(self.device)
                alpha_hat = self.alpha_hat[t][:, None, None, None].to(self.device)
                beta = self.beta[t][:, None, None, None].to(self.device)
                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
                # 중간 결과 시각화 및 저장
                if i % 110 == 0:
                    intermediate_images = (x.clamp(-1, 1) + 1) / 2
                    save_image(intermediate_images, f"output/sample_step_{i}.png")
                if i == 1:
                    intermediate_images = (x.clamp(-1, 1) + 1) / 2
                    save_image(intermediate_images, f"output/sample_step_001.png")
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train_with_classifier(
    diffusion, dataloader, optimizer, num_epochs, device, use_amp
):
    mse = nn.MSELoss()

    if use_amp:
        scaler = GradScaler()

    diffusion.train()

    if not os.path.exists("output"):
        os.makedirs("output", exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}")
        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch} / {num_epochs}", position=0, leave=True
        )
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)  # 레이블도 디바이스로 이동
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None

            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    predicted_noise = diffusion.model(x_t, t, labels)  # 레이블 추가
                    loss = mse(noise, predicted_noise)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predicted_noise = diffusion.model(x_t, t, labels)  # 레이블 추가
                loss = mse(noise, predicted_noise)
                loss.backward()
                optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        print(f"Epoch {epoch}: Loss = {loss.item()}")

        # Save model checkpoint
        checkpoint_path = f"./output/diffusion_unet_classifier_checkpoint.pth"
        torch.save(diffusion.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    torch.save(diffusion.model.state_dict(), "./diffusion_unet_classifier400_32.pth")
    print(f"Model saved to ./diffusion_unet_classifier.pth")

# 텍스트 프롬프트를 받아 해당 클래스의 인덱스를 반환하는 함수
def get_class_index(prompt, class_names):
    # CLIP 모델과 전처리기 로드
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # CIFAR-10 클래스 이름을 텍스트로 인코딩
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    
    # 텍스트 인코딩
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    # 프롬프트를 토큰화하고 인코딩
    text_prompt = clip.tokenize(prompt).to(device)
    with torch.no_grad():
        prompt_features = model.encode_text(text_prompt)
    
    # 유사도 계산
    similarities = (prompt_features @ text_features.T).squeeze()
    best_class = similarities.argmax().item()
    
    return best_class

def generate_with_prompt(diffusion, dataset, class_names, prompt, n_samples=16, img_size=32):
    target_class = get_class_index(prompt, class_names)
    
    original_images = [img for img, label in dataset if label == target_class]
    original_images = original_images[:n_samples]
    original_images = torch.stack(original_images).to(diffusion.device)

    resize_transform = transforms.Resize((img_size, img_size))
    unnormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    diffusion.eval()
    with torch.no_grad():
        sampled_images = diffusion.sample_with_classifier(
            diffusion.model, n=n_samples, target_class=target_class
        )
        sampled_images = sampled_images.cpu()
        sampled_images = torch.stack([resize_transform(img) for img in sampled_images])
        sampled_images = make_grid(sampled_images, nrow=6)
        sampled_images = sampled_images.permute(1, 2, 0).numpy()

    original_images = original_images.cpu()
    original_images = torch.stack([resize_transform(img) for img in original_images])
    original_images = torch.stack([unnormalize(img) for img in original_images])
    original_images = make_grid(original_images, nrow=6)
    original_images = original_images.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))

    axes[0].imshow(original_images)
    axes[0].axis("off")
    axes[0].set_title("Original Images")

    axes[1].imshow(sampled_images)
    axes[1].axis("off")
    axes[1].set_title("Generated Images")

    plt.show()

def generate_with_classifier(diffusion, dataset, n_samples=16, img_size=32, target_class=None):
    # CIFAR-10 데이터셋에서 target_class에 해당하는 원본 이미지를 선택
    original_images = [img for img, label in dataset if label == target_class]
    original_images = original_images[:n_samples]  # 선택한 이미지를 n_samples 만큼 제한
    original_images = torch.stack(original_images).to(diffusion.device)

    resize_transform = transforms.Resize((img_size, img_size))
    unnormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    diffusion.eval()
    with torch.no_grad():
        sampled_images = diffusion.sample_with_classifier(
            diffusion.model, n=n_samples, target_class=target_class
        )
        sampled_images = sampled_images.cpu()
        sampled_images = torch.stack([resize_transform(img) for img in sampled_images])
        sampled_images = make_grid(sampled_images, nrow=6)
        sampled_images = sampled_images.permute(1, 2, 0).numpy()

    # 원본 이미지들을 처리 (역정규화)
    original_images = original_images.cpu()
    original_images = torch.stack([resize_transform(img) for img in original_images])
    original_images = torch.stack([unnormalize(img) for img in original_images])
    original_images = make_grid(original_images, nrow=6)
    original_images = original_images.permute(1, 2, 0).numpy()

    # Display the collected images
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))

    axes[0].imshow(original_images)
    axes[0].axis("off")
    axes[0].set_title("Original Images")

    axes[1].imshow(sampled_images)
    axes[1].axis("off")
    axes[1].set_title("Generated Images")

    plt.show()


if __name__ == "__main__":

    # Hyperparameters
    batch_size = 32
    image_size = 32
    num_epochs = 400
    learning_rate = 1e-4
    train_model = False  # Set this to False to load a model and generate images

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
        use_amp = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
        use_amp = True
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
        use_amp = False

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./cifar10/train", train=True, download=True, transform=transform
    )

    # Extract the class names
    class_names = dataset.classes

    # Number of classes
    num_classes = len(class_names)

    # Print the results
    print(f"The CIFAR-10 dataset has {num_classes} classes.")
    print("The classes are:", class_names)

    # 10,000개의 데이터만 사용하도록 서브셋 생성
    indices = np.random.choice(len(dataset), 10000, replace=False)
    subset_dataset = Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )

    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")

    model = UNet_classifier(num_classes=10, device=device).to(device)
    diffusion = GaussianDiffusion(model, device=device, img_size=image_size).to(device)
    optimizer = optim.Adam(diffusion.parameters(), lr=learning_rate)

    if train_model:
        print("Starting training loop...")
        train_with_classifier(
            diffusion,
            dataloader,
            optimizer,
            num_epochs=num_epochs,
            device=device,
            use_amp=use_amp,
        )
    else:
        print("Loading saved model...")
        model.load_state_dict(
            torch.load("./diffusion_unet_classifier400_32.pth", map_location=device)
        )
        diffusion.model = model
        # generate_with_classifier(diffusion, dataset=dataset, n_samples=36, img_size=256, target_class=1)
        
        prompt = "A red car parked on the street"  # 예시 프롬프트
        generate_with_prompt(diffusion, dataset, class_names, prompt, n_samples=36, img_size=256)