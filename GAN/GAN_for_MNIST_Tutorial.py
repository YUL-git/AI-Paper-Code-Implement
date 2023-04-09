#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system("jupyter nbconvert --to script 'GAN.ipynb'")


# #### <b>GAN 실습</b>
# 
# * 논문 제목: Generative Adversarial Networks <b>(NIPS 2014)</b>
# * 가장 기본적인 GAN 모델을 학습해보는 실습을 진행합니다.
# * 학습 데이터셋: <b>MNIST</b> (1 X 28 X 28)

# #### <b>필요한 라이브러리 불러오기</b>
# 
# * 실습을 위한 PyTorch 라이브러리를 불러옵니다.

# In[1]:


import torch
import torch.nn as nn

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image


# #### <b>생성자(Generator) 및 판별자(Discriminator) 모델 정의</b>

# In[2]:


latent_dim = 100

# 생성자(Generator) 클래스 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 하나의 블록(block) 정의
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                # 배치 정규화(batch normalization) 수행(차원 동일)
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 생성자 모델은 연속적인 여러 개의 블록을 가짐
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


# In[3]:


# 판별자(Discriminator) 클래스 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    # 이미지에 대한 판별 결과를 반환
    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)

        return output


# #### <b>학습 데이터셋 불러오기</b>
# 
# * 학습을 위해 MNIST 데이터셋을 불러옵니다.

# In[4]:


transforms_train = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms_train)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)


# #### <b>모델 학습 및 샘플링</b>
# 
# * 학습을 위해 생성자와 판별자 모델을 초기화합니다.
# * 적절한 하이퍼 파라미터를 설정합니다.

# In[10]:


# 생성자(generator)와 판별자(discriminator) 초기화
generator = Generator()
discriminator = Discriminator()

generator.cuda()
discriminator.cuda()

# 손실 함수(loss function)
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

# 학습률(learning rate) 설정
lr = 0.0002

# 생성자와 판별자를 위한 최적화 함수
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


# * 모델을 학습하면서 주기적으로 샘플링하여 결과를 확인할 수 있습니다.

# In[ ]:


import time

n_epochs = 200 # 학습의 횟수(epoch) 설정
sample_interval = 2000 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정
start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # 진짜(real): 1
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # 가짜(fake): 0

        real_imgs = imgs.cuda()

        """ 생성자(generator)를 학습합니다. """
        optimizer_G.zero_grad()

        # 랜덤 노이즈(noise) 샘플링
        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()

        # 이미지 생성
        generated_imgs = generator(z)

        # 생성자(generator)의 손실(loss) 값 계산
        g_loss = adversarial_loss(discriminator(generated_imgs), real)

        # 생성자(generator) 업데이트
        g_loss.backward()
        optimizer_G.step()

        """ 판별자(discriminator)를 학습합니다. """
        optimizer_D.zero_grad()

        # 판별자(discriminator)의 손실(loss) 값 계산
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # 판별자(discriminator) 업데이트
        d_loss.backward()
        optimizer_D.step()

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:
            # 생성된 이미지 중에서 25개만 선택하여 5 X 5 격자 이미지에 출력
            save_image(generated_imgs.data[:25], f"{done}.png", nrow=5, normalize=True)

    # 하나의 epoch이 끝날 때마다 로그(log) 출력
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")


# * 생성된 이미지 예시를 출력합니다.

# In[4]:


from IPython.display import Image

Image('92000.png')


# 논문에서는 k = 1로 두었지만 D의 학습 양을 k > 1로 두고 학습 시키고 싶었습니다.

# In[ ]:


import time

n_epochs = 200 # Number of epochs
sample_interval = 2000 # Print results every `sample_interval` batches
k = 5 # Number of times to update the discriminator per generator update
start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Real and fake labels
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # Real: 1
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # Fake: 0

        real_imgs = imgs.cuda()

        """ Update the discriminator k times. """
        for j in range(k):
            optimizer_D.zero_grad()

            # Discriminator loss for real images
            real_loss = adversarial_loss(discriminator(real_imgs), real)

            # Discriminator loss for fake images
            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()
            generated_imgs = generator(z).detach() # Detach to avoid computing gradients for the generator
            fake_loss = adversarial_loss(discriminator(generated_imgs), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            # Update the discriminator
            d_loss.backward()
            optimizer_D.step()

        """ Update the generator once. """
        optimizer_G.zero_grad()

        # Generator loss
        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()
        generated_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(generated_imgs), real)

        # Update the generator
        g_loss.backward()
        optimizer_G.step()

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:
            # Save 25 generated images as a 5x5 grid
            save_image(generated_imgs.data[:25], f"{done}.png", nrow=5, normalize=True)

    # Print log at the end of each epoch
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")

