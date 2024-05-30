import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os
import psutil
import torch.nn.functional as F

# 시드 고정
manualSeed = 999
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manualSeed)

# 메모리 사용량 확인 함수
def check_memory_usage():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()}")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()}")
    print(f"System Memory Used: {psutil.virtual_memory().percent}%")
    
def self_attention(x):
    batch_size, C, width, height = x.size()
    query_conv = nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1).to(x.device)
    key_conv = nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1).to(x.device)
    value_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1).to(x.device)
    gamma = nn.Parameter(torch.zeros(1)).to(x.device)

    proj_query = query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
    proj_key = key_conv(x).view(batch_size, -1, width * height)
    energy = torch.bmm(proj_query, proj_key)
    attention = F.softmax(energy, dim=-1)
    proj_value = value_conv(x).view(batch_size, -1, width * height)
    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(batch_size, C, width, height)
    out = gamma * out + x
    return out

# GAN 생성자 정의
def create_generator(nz):
    generator = nn.Sequential(
        nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
        nn.LayerNorm([512, 4, 4]),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        nn.LayerNorm([256, 8, 8]),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.LayerNorm([128, 16, 16]),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.LayerNorm([64, 32, 32]),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    return generator

def forward_generator(x, generator):
    x = generator[0:7](x)
    x = self_attention(x)  # Self-Attention 적용
    x = generator[7:](x)
    return x

# GAN 판별자 정의
def create_discriminator():
    discriminator = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        nn.LayerNorm([128, 16, 16]),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1, bias=False),
        nn.LayerNorm([256, 8, 8]),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        nn.LayerNorm([512, 4, 4]),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        nn.Flatten()
    )
    return discriminator

def forward_discriminator(x, discriminator):
    x = discriminator[0:7](x)
    x = self_attention(x)  # Self-Attention 적용
    x = discriminator[7:](x)
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = interpolated.requires_grad_(True)

    prob_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty

def train_gan(num_epochs, dataloader, nz, device, batch_size):
    netG = create_generator(nz).to(device)
    netD = create_discriminator().to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    G_losses = []
    D_losses = []
    gp_lambda = 5
    n_critic = 3
    num_row = 8
    learning_rate = 0.00001
    
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    
    real_batch = next(iter(dataloader))[0].to(device)
    axs[0].imshow(np.transpose(vutils.make_grid(real_batch[:batch_size], nrow=num_row, padding=5, normalize=True).cpu(), (1, 2, 0)))
    axs[0].set_title("Real Images")
    axs[0].axis("off")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            try:
                netD.zero_grad()
                real = data[0].to(device)
                batch_size = real.size(0)
                
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = forward_generator(noise, netG)  # 수정된 부분
                
                output_real = forward_discriminator(real, netD).view(-1)  # 수정된 부분
                output_fake = forward_discriminator(fake.detach(), netD).view(-1)  # 수정된 부분
                gradient_pen = gradient_penalty(netD, real, fake.detach(), device)
                errD = -torch.mean(output_real) + torch.mean(output_fake) + gp_lambda * gradient_pen
                errD.backward()
                optimizerD.step()
                
                D_losses.append(errD.item())

                if i % n_critic == 0:
                    netG.zero_grad()
                    output_fake = forward_discriminator(fake, netD).view(-1)  # 수정된 부분
                    errG = -torch.mean(output_fake)
                    errG.backward()
                    optimizerG.step()
                
                    G_losses.append(errG.item())

                if i % 100 == 0:
                    print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')
                    vutils.save_image(fake.detach(), f'{output_dir}/fake_samples_epoch_{epoch}_batch_{i}.png', nrow=num_row, normalize=True)

                    # 생성된 이미지 시각화
                    with torch.no_grad():
                        fake = forward_generator(fixed_noise, netG).detach().cpu()  # 수정된 부분
                    axs[1].clear()
                    axs[1].imshow(np.transpose(vutils.make_grid(fake, nrow=num_row, padding=5, normalize=True), (1, 2, 0)))
                    axs[1].set_title("Fake Images")
                    axs[1].axis("off")
                    plt.pause(0.001)
            except Exception as e:
                print(f'Error at epoch {epoch}, batch {i}: {str(e)}')
                continue

    plt.ioff()
    plt.show()
    
    return netG, netD



def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
    nz = 100  # 잠재 공간 벡터 크기
    num_epochs = 25
    batch_size = 64
        
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
        
    dataset = torchvision.datasets.CIFAR10(
        root="./cifar10/train", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    # 데이터셋의 모든 배치를 미리 검사하는 코드 추가
    for i, data in enumerate(dataloader, 0):
        try:
            inputs, labels = data
        except Exception as e:
            print(f'Error in batch {i}: {str(e)}')
    
    netG, netD = train_gan(num_epochs, dataloader, nz, device, batch_size)
        
if __name__ == "__main__":
    main()
