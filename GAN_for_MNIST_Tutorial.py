{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xrUqXYEepD0o"
   },
   "source": [
    "#### <b>GAN 실습</b>\n",
    "\n",
    "* 논문 제목: Generative Adversarial Networks <b>(NIPS 2014)</b>\n",
    "* 가장 기본적인 GAN 모델을 학습해보는 실습을 진행합니다.\n",
    "* 학습 데이터셋: <b>MNIST</b> (1 X 28 X 28)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "UEx96DYOpdAK"
   },
   "source": [
    "#### <b>필요한 라이브러리 불러오기</b>\n",
    "\n",
    "* 실습을 위한 PyTorch 라이브러리를 불러옵니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-04-09T11:08:32.806035Z",
     "iopub.status.busy": "2023-04-09T11:08:32.805650Z",
     "iopub.status.idle": "2023-04-09T11:08:33.876370Z",
     "shell.execute_reply": "2023-04-09T11:08:33.875240Z",
     "shell.execute_reply.started": "2023-04-09T11:08:32.806007Z"
    },
    "id": "CiRb7M3naHyo"
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "from torchvision import datasets\n",
    "import torchvision.transforms as transforms\n",
    "from torchvision.utils import save_image"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tp4hbA95pihv"
   },
   "source": [
    "#### <b>생성자(Generator) 및 판별자(Discriminator) 모델 정의</b>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-04-09T11:08:40.491332Z",
     "iopub.status.busy": "2023-04-09T11:08:40.490547Z",
     "iopub.status.idle": "2023-04-09T11:08:40.498453Z",
     "shell.execute_reply": "2023-04-09T11:08:40.497617Z",
     "shell.execute_reply.started": "2023-04-09T11:08:40.491304Z"
    },
    "id": "Hj5al6cTZES1"
   },
   "outputs": [],
   "source": [
    "latent_dim = 100\n",
    "\n",
    "# 생성자(Generator) 클래스 정의\n",
    "class Generator(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Generator, self).__init__()\n",
    "\n",
    "        # 하나의 블록(block) 정의\n",
    "        def block(input_dim, output_dim, normalize=True):\n",
    "            layers = [nn.Linear(input_dim, output_dim)]\n",
    "            if normalize:\n",
    "                # 배치 정규화(batch normalization) 수행(차원 동일)\n",
    "                layers.append(nn.BatchNorm1d(output_dim, 0.8))\n",
    "            layers.append(nn.LeakyReLU(0.2, inplace=True))\n",
    "            return layers\n",
    "\n",
    "        # 생성자 모델은 연속적인 여러 개의 블록을 가짐\n",
    "        self.model = nn.Sequential(\n",
    "            *block(latent_dim, 128, normalize=False),\n",
    "            *block(128, 256),\n",
    "            *block(256, 512),\n",
    "            *block(512, 1024),\n",
    "            nn.Linear(1024, 1 * 28 * 28),\n",
    "            nn.Tanh()\n",
    "        )\n",
    "\n",
    "    def forward(self, z):\n",
    "        img = self.model(z)\n",
    "        img = img.view(img.size(0), 1, 28, 28)\n",
    "        return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-04-09T11:08:43.788090Z",
     "iopub.status.busy": "2023-04-09T11:08:43.787613Z",
     "iopub.status.idle": "2023-04-09T11:08:43.794997Z",
     "shell.execute_reply": "2023-04-09T11:08:43.793931Z",
     "shell.execute_reply.started": "2023-04-09T11:08:43.788055Z"
    },
    "id": "M_kvtvOhaLX6"
   },
   "outputs": [],
   "source": [
    "# 판별자(Discriminator) 클래스 정의\n",
    "class Discriminator(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Discriminator, self).__init__()\n",
    "\n",
    "        self.model = nn.Sequential(\n",
    "            nn.Linear(1 * 28 * 28, 512),\n",
    "            nn.LeakyReLU(0.2, inplace=True),\n",
    "            nn.Linear(512, 256),\n",
    "            nn.LeakyReLU(0.2, inplace=True),\n",
    "            nn.Linear(256, 1),\n",
    "            nn.Sigmoid(),\n",
    "        )\n",
    "\n",
    "    # 이미지에 대한 판별 결과를 반환\n",
    "    def forward(self, img):\n",
    "        flattened = img.view(img.size(0), -1)\n",
    "        output = self.model(flattened)\n",
    "\n",
    "        return output"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NOilX0rBqJXn"
   },
   "source": [
    "#### <b>학습 데이터셋 불러오기</b>\n",
    "\n",
    "* 학습을 위해 MNIST 데이터셋을 불러옵니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-04-09T08:22:58.330901Z",
     "iopub.status.busy": "2023-04-09T08:22:58.330308Z",
     "iopub.status.idle": "2023-04-09T08:23:04.171636Z",
     "shell.execute_reply": "2023-04-09T08:23:04.171015Z",
     "shell.execute_reply.started": "2023-04-09T08:22:58.330870Z"
    },
    "id": "HrhXIwtAqM7H"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./dataset/MNIST/raw/train-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5a223c6ea2314f7796924e2fe25ef7f6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/9912422 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./dataset/MNIST/raw/train-images-idx3-ubyte.gz to ./dataset/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./dataset/MNIST/raw/train-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2f67f8c022d8432b8d10f6e160688f9d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/28881 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./dataset/MNIST/raw/train-labels-idx1-ubyte.gz to ./dataset/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./dataset/MNIST/raw/t10k-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7fd0f52a8e824e8ca435d79c03ce7f10",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/1648877 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./dataset/MNIST/raw/t10k-images-idx3-ubyte.gz to ./dataset/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "43305f10561c4bdfbe1dc0fcb0960eb7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/4542 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./dataset/MNIST/raw\n",
      "\n"
     ]
    }
   ],
   "source": [
    "transforms_train = transforms.Compose([\n",
    "    transforms.Resize(28),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize([0.5], [0.5])\n",
    "])\n",
    "\n",
    "train_dataset = datasets.MNIST(root=\"./dataset\", train=True, download=True, transform=transforms_train)\n",
    "dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "K54Z7PNIqTkO"
   },
   "source": [
    "#### <b>모델 학습 및 샘플링</b>\n",
    "\n",
    "* 학습을 위해 생성자와 판별자 모델을 초기화합니다.\n",
    "* 적절한 하이퍼 파라미터를 설정합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-04-09T08:23:42.569597Z",
     "iopub.status.busy": "2023-04-09T08:23:42.569068Z",
     "iopub.status.idle": "2023-04-09T08:23:43.814354Z",
     "shell.execute_reply": "2023-04-09T08:23:43.813823Z",
     "shell.execute_reply.started": "2023-04-09T08:23:42.569573Z"
    },
    "id": "tBZf0BmBaN7l"
   },
   "outputs": [],
   "source": [
    "# 생성자(generator)와 판별자(discriminator) 초기화\n",
    "generator = Generator()\n",
    "discriminator = Discriminator()\n",
    "\n",
    "generator.cuda()\n",
    "discriminator.cuda()\n",
    "\n",
    "# 손실 함수(loss function)\n",
    "adversarial_loss = nn.BCELoss()\n",
    "adversarial_loss.cuda()\n",
    "\n",
    "# 학습률(learning rate) 설정\n",
    "lr = 0.0002\n",
    "\n",
    "# 생성자와 판별자를 위한 최적화 함수\n",
    "optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))\n",
    "optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "F9ThAQIOt-74"
   },
   "source": [
    "* 모델을 학습하면서 주기적으로 샘플링하여 결과를 확인할 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "srQI5xI6ar-X",
    "outputId": "0bd2c30f-245a-4dea-8660-5a4c927de52e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 0/200] [D loss: 0.522748] [G loss: 0.829430] [Elapsed time: 14.64s]\n",
      "[Epoch 1/200] [D loss: 0.372858] [G loss: 1.517679] [Elapsed time: 29.62s]\n",
      "[Epoch 2/200] [D loss: 0.363016] [G loss: 1.329226] [Elapsed time: 44.35s]\n",
      "[Epoch 3/200] [D loss: 0.312027] [G loss: 1.289821] [Elapsed time: 58.97s]\n",
      "[Epoch 4/200] [D loss: 0.711739] [G loss: 0.354931] [Elapsed time: 73.76s]\n",
      "[Epoch 5/200] [D loss: 0.251453] [G loss: 1.115114] [Elapsed time: 88.81s]\n",
      "[Epoch 6/200] [D loss: 0.222200] [G loss: 1.638794] [Elapsed time: 103.76s]\n",
      "[Epoch 7/200] [D loss: 0.212449] [G loss: 1.299674] [Elapsed time: 118.63s]\n",
      "[Epoch 8/200] [D loss: 0.188896] [G loss: 1.600519] [Elapsed time: 133.08s]\n",
      "[Epoch 9/200] [D loss: 0.226206] [G loss: 2.084917] [Elapsed time: 148.05s]\n",
      "[Epoch 10/200] [D loss: 0.225342] [G loss: 2.228838] [Elapsed time: 163.09s]\n",
      "[Epoch 11/200] [D loss: 0.276130] [G loss: 1.518153] [Elapsed time: 178.27s]\n",
      "[Epoch 12/200] [D loss: 0.216676] [G loss: 1.424551] [Elapsed time: 193.14s]\n",
      "[Epoch 13/200] [D loss: 0.250011] [G loss: 1.505254] [Elapsed time: 207.70s]\n",
      "[Epoch 14/200] [D loss: 0.185601] [G loss: 3.071321] [Elapsed time: 222.36s]\n",
      "[Epoch 15/200] [D loss: 0.234813] [G loss: 1.806695] [Elapsed time: 237.22s]\n",
      "[Epoch 16/200] [D loss: 0.249854] [G loss: 1.138674] [Elapsed time: 251.82s]\n",
      "[Epoch 17/200] [D loss: 0.249046] [G loss: 4.156915] [Elapsed time: 266.81s]\n",
      "[Epoch 18/200] [D loss: 0.321985] [G loss: 3.767182] [Elapsed time: 281.35s]\n",
      "[Epoch 19/200] [D loss: 0.312805] [G loss: 0.984120] [Elapsed time: 296.21s]\n",
      "[Epoch 20/200] [D loss: 0.133047] [G loss: 2.177111] [Elapsed time: 311.03s]\n",
      "[Epoch 21/200] [D loss: 0.287964] [G loss: 1.124958] [Elapsed time: 325.98s]\n",
      "[Epoch 22/200] [D loss: 0.143492] [G loss: 2.103356] [Elapsed time: 340.97s]\n",
      "[Epoch 23/200] [D loss: 0.697677] [G loss: 4.264883] [Elapsed time: 355.85s]\n",
      "[Epoch 24/200] [D loss: 0.230640] [G loss: 1.688197] [Elapsed time: 370.62s]\n",
      "[Epoch 25/200] [D loss: 0.375375] [G loss: 4.324008] [Elapsed time: 385.57s]\n",
      "[Epoch 26/200] [D loss: 0.181044] [G loss: 2.331903] [Elapsed time: 400.26s]\n",
      "[Epoch 27/200] [D loss: 0.147811] [G loss: 2.690580] [Elapsed time: 415.07s]\n",
      "[Epoch 28/200] [D loss: 0.215825] [G loss: 3.242933] [Elapsed time: 430.07s]\n",
      "[Epoch 29/200] [D loss: 0.361689] [G loss: 3.526228] [Elapsed time: 445.25s]\n",
      "[Epoch 30/200] [D loss: 0.262714] [G loss: 4.088052] [Elapsed time: 460.18s]\n",
      "[Epoch 31/200] [D loss: 0.389835] [G loss: 0.873464] [Elapsed time: 475.16s]\n",
      "[Epoch 32/200] [D loss: 0.200233] [G loss: 2.727276] [Elapsed time: 490.73s]\n",
      "[Epoch 33/200] [D loss: 0.344501] [G loss: 3.287966] [Elapsed time: 505.74s]\n",
      "[Epoch 34/200] [D loss: 1.131307] [G loss: 7.970099] [Elapsed time: 520.81s]\n",
      "[Epoch 35/200] [D loss: 0.210656] [G loss: 1.844736] [Elapsed time: 536.34s]\n",
      "[Epoch 36/200] [D loss: 0.380815] [G loss: 4.722221] [Elapsed time: 551.15s]\n",
      "[Epoch 37/200] [D loss: 0.204039] [G loss: 1.780442] [Elapsed time: 566.02s]\n",
      "[Epoch 38/200] [D loss: 0.126606] [G loss: 3.045232] [Elapsed time: 581.24s]\n",
      "[Epoch 39/200] [D loss: 0.164484] [G loss: 2.620150] [Elapsed time: 596.09s]\n",
      "[Epoch 40/200] [D loss: 0.417071] [G loss: 0.942508] [Elapsed time: 610.90s]\n",
      "[Epoch 41/200] [D loss: 0.344890] [G loss: 1.239429] [Elapsed time: 625.52s]\n",
      "[Epoch 42/200] [D loss: 0.219357] [G loss: 1.801792] [Elapsed time: 640.42s]\n",
      "[Epoch 43/200] [D loss: 0.261628] [G loss: 1.403077] [Elapsed time: 655.52s]\n",
      "[Epoch 44/200] [D loss: 0.264856] [G loss: 1.670803] [Elapsed time: 669.95s]\n",
      "[Epoch 45/200] [D loss: 0.320740] [G loss: 1.294596] [Elapsed time: 684.61s]\n",
      "[Epoch 46/200] [D loss: 0.735896] [G loss: 0.457015] [Elapsed time: 699.22s]\n",
      "[Epoch 47/200] [D loss: 0.215527] [G loss: 1.622929] [Elapsed time: 714.20s]\n",
      "[Epoch 48/200] [D loss: 0.310391] [G loss: 1.838112] [Elapsed time: 728.92s]\n",
      "[Epoch 49/200] [D loss: 0.326488] [G loss: 1.143378] [Elapsed time: 744.23s]\n",
      "[Epoch 50/200] [D loss: 0.488925] [G loss: 3.819730] [Elapsed time: 759.78s]\n",
      "[Epoch 51/200] [D loss: 0.281829] [G loss: 1.752762] [Elapsed time: 774.57s]\n",
      "[Epoch 52/200] [D loss: 0.293825] [G loss: 1.591572] [Elapsed time: 789.61s]\n",
      "[Epoch 53/200] [D loss: 0.305799] [G loss: 1.273533] [Elapsed time: 804.35s]\n",
      "[Epoch 54/200] [D loss: 0.363520] [G loss: 2.063176] [Elapsed time: 819.70s]\n",
      "[Epoch 55/200] [D loss: 0.280371] [G loss: 1.596596] [Elapsed time: 834.28s]\n",
      "[Epoch 56/200] [D loss: 0.332533] [G loss: 2.254220] [Elapsed time: 849.09s]\n",
      "[Epoch 57/200] [D loss: 0.323975] [G loss: 2.045448] [Elapsed time: 863.83s]\n",
      "[Epoch 58/200] [D loss: 0.312644] [G loss: 1.940239] [Elapsed time: 878.52s]\n",
      "[Epoch 59/200] [D loss: 0.433621] [G loss: 0.831792] [Elapsed time: 892.98s]\n",
      "[Epoch 60/200] [D loss: 0.379544] [G loss: 1.479267] [Elapsed time: 907.94s]\n",
      "[Epoch 61/200] [D loss: 0.337388] [G loss: 2.652977] [Elapsed time: 923.25s]\n",
      "[Epoch 62/200] [D loss: 0.289478] [G loss: 1.463956] [Elapsed time: 938.16s]\n",
      "[Epoch 63/200] [D loss: 0.332678] [G loss: 1.354259] [Elapsed time: 953.27s]\n",
      "[Epoch 64/200] [D loss: 0.497146] [G loss: 0.947130] [Elapsed time: 968.09s]\n",
      "[Epoch 65/200] [D loss: 0.298476] [G loss: 1.794587] [Elapsed time: 982.46s]\n",
      "[Epoch 66/200] [D loss: 0.342376] [G loss: 2.761930] [Elapsed time: 997.41s]\n",
      "[Epoch 67/200] [D loss: 0.199975] [G loss: 2.286337] [Elapsed time: 1012.32s]\n",
      "[Epoch 68/200] [D loss: 0.550259] [G loss: 4.399807] [Elapsed time: 1026.82s]\n",
      "[Epoch 69/200] [D loss: 0.348959] [G loss: 1.681570] [Elapsed time: 1041.37s]\n",
      "[Epoch 70/200] [D loss: 0.350790] [G loss: 2.678944] [Elapsed time: 1055.61s]\n",
      "[Epoch 71/200] [D loss: 0.349744] [G loss: 1.321077] [Elapsed time: 1070.66s]\n",
      "[Epoch 72/200] [D loss: 0.454832] [G loss: 0.855591] [Elapsed time: 1085.41s]\n",
      "[Epoch 73/200] [D loss: 0.216424] [G loss: 2.202238] [Elapsed time: 1100.12s]\n",
      "[Epoch 74/200] [D loss: 0.332032] [G loss: 1.467984] [Elapsed time: 1114.63s]\n",
      "[Epoch 75/200] [D loss: 0.334628] [G loss: 1.495943] [Elapsed time: 1129.88s]\n",
      "[Epoch 76/200] [D loss: 0.275214] [G loss: 1.497794] [Elapsed time: 1144.08s]\n",
      "[Epoch 77/200] [D loss: 0.420321] [G loss: 0.879592] [Elapsed time: 1158.38s]\n",
      "[Epoch 78/200] [D loss: 0.406984] [G loss: 1.116465] [Elapsed time: 1172.16s]\n",
      "[Epoch 79/200] [D loss: 0.306117] [G loss: 1.586033] [Elapsed time: 1185.82s]\n",
      "[Epoch 80/200] [D loss: 0.346746] [G loss: 2.917901] [Elapsed time: 1199.78s]\n",
      "[Epoch 81/200] [D loss: 0.414805] [G loss: 4.355955] [Elapsed time: 1213.76s]\n",
      "[Epoch 82/200] [D loss: 0.217584] [G loss: 2.185797] [Elapsed time: 1228.37s]\n",
      "[Epoch 83/200] [D loss: 0.299525] [G loss: 3.134455] [Elapsed time: 1242.65s]\n",
      "[Epoch 84/200] [D loss: 0.453676] [G loss: 3.047255] [Elapsed time: 1257.05s]\n",
      "[Epoch 85/200] [D loss: 0.265712] [G loss: 2.504372] [Elapsed time: 1271.58s]\n",
      "[Epoch 86/200] [D loss: 0.468562] [G loss: 3.420906] [Elapsed time: 1286.16s]\n",
      "[Epoch 87/200] [D loss: 0.313388] [G loss: 3.086492] [Elapsed time: 1300.33s]\n",
      "[Epoch 88/200] [D loss: 0.286210] [G loss: 1.856419] [Elapsed time: 1314.43s]\n",
      "[Epoch 89/200] [D loss: 0.284829] [G loss: 2.903432] [Elapsed time: 1328.78s]\n",
      "[Epoch 90/200] [D loss: 0.334503] [G loss: 1.819413] [Elapsed time: 1342.66s]\n",
      "[Epoch 91/200] [D loss: 0.208366] [G loss: 2.947707] [Elapsed time: 1356.72s]\n",
      "[Epoch 92/200] [D loss: 0.339319] [G loss: 2.177942] [Elapsed time: 1371.20s]\n",
      "[Epoch 93/200] [D loss: 0.308104] [G loss: 2.216433] [Elapsed time: 1385.47s]\n",
      "[Epoch 94/200] [D loss: 0.203005] [G loss: 2.104164] [Elapsed time: 1399.74s]\n",
      "[Epoch 95/200] [D loss: 0.281514] [G loss: 2.535281] [Elapsed time: 1414.42s]\n",
      "[Epoch 96/200] [D loss: 0.323359] [G loss: 1.414945] [Elapsed time: 1428.69s]\n",
      "[Epoch 97/200] [D loss: 0.209258] [G loss: 2.121124] [Elapsed time: 1443.10s]\n",
      "[Epoch 98/200] [D loss: 0.211430] [G loss: 2.274003] [Elapsed time: 1457.63s]\n",
      "[Epoch 99/200] [D loss: 0.248021] [G loss: 2.419493] [Elapsed time: 1472.17s]\n",
      "[Epoch 100/200] [D loss: 0.283141] [G loss: 2.182565] [Elapsed time: 1486.11s]\n",
      "[Epoch 101/200] [D loss: 0.291069] [G loss: 1.625459] [Elapsed time: 1500.31s]\n",
      "[Epoch 102/200] [D loss: 0.266260] [G loss: 1.748940] [Elapsed time: 1514.75s]\n",
      "[Epoch 103/200] [D loss: 0.279893] [G loss: 1.801305] [Elapsed time: 1528.98s]\n",
      "[Epoch 104/200] [D loss: 0.337647] [G loss: 1.372875] [Elapsed time: 1543.34s]\n",
      "[Epoch 105/200] [D loss: 0.372733] [G loss: 2.310965] [Elapsed time: 1557.47s]\n",
      "[Epoch 106/200] [D loss: 0.304111] [G loss: 1.503514] [Elapsed time: 1571.95s]\n",
      "[Epoch 107/200] [D loss: 0.288937] [G loss: 2.483202] [Elapsed time: 1586.13s]\n",
      "[Epoch 108/200] [D loss: 0.294476] [G loss: 1.531241] [Elapsed time: 1600.49s]\n",
      "[Epoch 109/200] [D loss: 0.341307] [G loss: 2.055195] [Elapsed time: 1614.86s]\n",
      "[Epoch 110/200] [D loss: 0.322244] [G loss: 2.585032] [Elapsed time: 1629.25s]\n",
      "[Epoch 111/200] [D loss: 0.331668] [G loss: 1.277997] [Elapsed time: 1643.81s]\n",
      "[Epoch 112/200] [D loss: 0.257002] [G loss: 1.585161] [Elapsed time: 1658.57s]\n",
      "[Epoch 113/200] [D loss: 0.306509] [G loss: 1.389385] [Elapsed time: 1673.70s]\n",
      "[Epoch 114/200] [D loss: 0.288539] [G loss: 2.388597] [Elapsed time: 1689.25s]\n",
      "[Epoch 115/200] [D loss: 0.219805] [G loss: 2.253654] [Elapsed time: 1703.90s]\n",
      "[Epoch 116/200] [D loss: 0.251945] [G loss: 1.817379] [Elapsed time: 1718.33s]\n",
      "[Epoch 117/200] [D loss: 0.266896] [G loss: 2.087219] [Elapsed time: 1733.40s]\n",
      "[Epoch 118/200] [D loss: 0.239400] [G loss: 1.614851] [Elapsed time: 1748.42s]\n",
      "[Epoch 119/200] [D loss: 0.264902] [G loss: 2.895990] [Elapsed time: 1763.09s]\n",
      "[Epoch 120/200] [D loss: 0.297452] [G loss: 2.588012] [Elapsed time: 1777.57s]\n",
      "[Epoch 121/200] [D loss: 0.297142] [G loss: 2.225440] [Elapsed time: 1792.15s]\n",
      "[Epoch 122/200] [D loss: 0.268688] [G loss: 1.726179] [Elapsed time: 1806.62s]\n",
      "[Epoch 123/200] [D loss: 0.310022] [G loss: 1.661120] [Elapsed time: 1821.90s]\n",
      "[Epoch 124/200] [D loss: 0.144470] [G loss: 2.571463] [Elapsed time: 1836.72s]\n",
      "[Epoch 125/200] [D loss: 0.205969] [G loss: 2.516297] [Elapsed time: 1851.07s]\n",
      "[Epoch 126/200] [D loss: 0.327357] [G loss: 2.757601] [Elapsed time: 1866.09s]\n",
      "[Epoch 127/200] [D loss: 0.269763] [G loss: 1.721026] [Elapsed time: 1881.54s]\n",
      "[Epoch 128/200] [D loss: 0.221831] [G loss: 2.731440] [Elapsed time: 1896.85s]\n",
      "[Epoch 129/200] [D loss: 0.321697] [G loss: 2.349569] [Elapsed time: 1911.31s]\n",
      "[Epoch 130/200] [D loss: 0.323291] [G loss: 1.700038] [Elapsed time: 1925.97s]\n",
      "[Epoch 131/200] [D loss: 0.265377] [G loss: 1.847327] [Elapsed time: 1940.73s]\n",
      "[Epoch 132/200] [D loss: 0.308373] [G loss: 1.569885] [Elapsed time: 1955.58s]\n",
      "[Epoch 133/200] [D loss: 0.208499] [G loss: 2.428175] [Elapsed time: 1970.32s]\n",
      "[Epoch 134/200] [D loss: 0.305512] [G loss: 1.943389] [Elapsed time: 1984.97s]\n",
      "[Epoch 135/200] [D loss: 0.272931] [G loss: 2.183786] [Elapsed time: 2000.05s]\n",
      "[Epoch 136/200] [D loss: 0.359679] [G loss: 2.021343] [Elapsed time: 2015.21s]\n",
      "[Epoch 137/200] [D loss: 0.357651] [G loss: 1.517780] [Elapsed time: 2030.43s]\n",
      "[Epoch 138/200] [D loss: 0.184322] [G loss: 2.807851] [Elapsed time: 2045.27s]\n",
      "[Epoch 139/200] [D loss: 0.203373] [G loss: 2.053690] [Elapsed time: 2060.09s]\n",
      "[Epoch 140/200] [D loss: 0.243935] [G loss: 1.716758] [Elapsed time: 2075.23s]\n",
      "[Epoch 141/200] [D loss: 0.322379] [G loss: 2.545387] [Elapsed time: 2090.06s]\n",
      "[Epoch 142/200] [D loss: 0.229060] [G loss: 1.970968] [Elapsed time: 2104.85s]\n",
      "[Epoch 143/200] [D loss: 0.274554] [G loss: 1.589352] [Elapsed time: 2119.89s]\n",
      "[Epoch 144/200] [D loss: 0.288916] [G loss: 2.178538] [Elapsed time: 2134.73s]\n",
      "[Epoch 145/200] [D loss: 0.268774] [G loss: 1.766644] [Elapsed time: 2149.62s]\n",
      "[Epoch 146/200] [D loss: 0.289206] [G loss: 2.783231] [Elapsed time: 2164.31s]\n",
      "[Epoch 147/200] [D loss: 0.225266] [G loss: 1.906159] [Elapsed time: 2179.32s]\n",
      "[Epoch 148/200] [D loss: 0.294216] [G loss: 2.146295] [Elapsed time: 2194.25s]\n",
      "[Epoch 149/200] [D loss: 0.360563] [G loss: 4.088191] [Elapsed time: 2209.58s]\n",
      "[Epoch 150/200] [D loss: 0.265889] [G loss: 2.245625] [Elapsed time: 2224.41s]\n",
      "[Epoch 151/200] [D loss: 0.291742] [G loss: 1.483999] [Elapsed time: 2239.37s]\n",
      "[Epoch 152/200] [D loss: 0.341779] [G loss: 1.542900] [Elapsed time: 2254.13s]\n",
      "[Epoch 153/200] [D loss: 0.200704] [G loss: 2.805265] [Elapsed time: 2268.69s]\n",
      "[Epoch 154/200] [D loss: 0.295421] [G loss: 2.333511] [Elapsed time: 2283.07s]\n",
      "[Epoch 155/200] [D loss: 0.296589] [G loss: 2.012913] [Elapsed time: 2297.71s]\n",
      "[Epoch 156/200] [D loss: 0.275950] [G loss: 2.416452] [Elapsed time: 2312.39s]\n",
      "[Epoch 157/200] [D loss: 0.213738] [G loss: 2.616714] [Elapsed time: 2327.23s]\n",
      "[Epoch 158/200] [D loss: 0.241508] [G loss: 2.677406] [Elapsed time: 2341.96s]\n",
      "[Epoch 159/200] [D loss: 0.194161] [G loss: 2.046714] [Elapsed time: 2356.83s]\n",
      "[Epoch 160/200] [D loss: 0.260721] [G loss: 1.942953] [Elapsed time: 2371.48s]\n",
      "[Epoch 161/200] [D loss: 0.314604] [G loss: 2.010342] [Elapsed time: 2386.16s]\n",
      "[Epoch 162/200] [D loss: 0.210261] [G loss: 1.962100] [Elapsed time: 2401.35s]\n",
      "[Epoch 163/200] [D loss: 0.248817] [G loss: 1.703924] [Elapsed time: 2416.14s]\n",
      "[Epoch 164/200] [D loss: 0.295533] [G loss: 2.154848] [Elapsed time: 2431.00s]\n",
      "[Epoch 165/200] [D loss: 0.360170] [G loss: 1.186311] [Elapsed time: 2445.93s]\n",
      "[Epoch 166/200] [D loss: 0.246468] [G loss: 2.128010] [Elapsed time: 2460.56s]\n",
      "[Epoch 167/200] [D loss: 0.267691] [G loss: 1.717979] [Elapsed time: 2475.68s]\n",
      "[Epoch 168/200] [D loss: 0.238555] [G loss: 2.982985] [Elapsed time: 2490.28s]\n",
      "[Epoch 169/200] [D loss: 0.299880] [G loss: 1.650179] [Elapsed time: 2504.60s]\n",
      "[Epoch 170/200] [D loss: 0.286263] [G loss: 1.650552] [Elapsed time: 2520.00s]\n",
      "[Epoch 171/200] [D loss: 0.351728] [G loss: 1.566897] [Elapsed time: 2535.14s]\n",
      "[Epoch 172/200] [D loss: 0.289074] [G loss: 2.197316] [Elapsed time: 2549.66s]\n",
      "[Epoch 173/200] [D loss: 0.299471] [G loss: 2.540869] [Elapsed time: 2564.27s]\n",
      "[Epoch 174/200] [D loss: 0.262938] [G loss: 2.057458] [Elapsed time: 2578.87s]\n",
      "[Epoch 175/200] [D loss: 0.324657] [G loss: 2.369673] [Elapsed time: 2593.65s]\n",
      "[Epoch 176/200] [D loss: 0.274170] [G loss: 1.926636] [Elapsed time: 2608.21s]\n",
      "[Epoch 177/200] [D loss: 0.242073] [G loss: 2.631045] [Elapsed time: 2623.69s]\n",
      "[Epoch 178/200] [D loss: 0.304828] [G loss: 1.962924] [Elapsed time: 2638.48s]\n",
      "[Epoch 179/200] [D loss: 0.309665] [G loss: 2.774096] [Elapsed time: 2653.47s]\n",
      "[Epoch 180/200] [D loss: 0.436067] [G loss: 1.677492] [Elapsed time: 2668.45s]\n",
      "[Epoch 181/200] [D loss: 0.208289] [G loss: 2.460046] [Elapsed time: 2683.23s]\n",
      "[Epoch 182/200] [D loss: 0.464152] [G loss: 4.730225] [Elapsed time: 2697.73s]\n",
      "[Epoch 183/200] [D loss: 0.236137] [G loss: 2.024809] [Elapsed time: 2712.46s]\n",
      "[Epoch 184/200] [D loss: 0.280529] [G loss: 2.612814] [Elapsed time: 2727.13s]\n",
      "[Epoch 185/200] [D loss: 0.373649] [G loss: 3.094165] [Elapsed time: 2741.85s]\n",
      "[Epoch 186/200] [D loss: 0.244470] [G loss: 2.152431] [Elapsed time: 2756.89s]\n",
      "[Epoch 187/200] [D loss: 0.262233] [G loss: 2.924129] [Elapsed time: 2771.52s]\n",
      "[Epoch 188/200] [D loss: 0.210062] [G loss: 2.471434] [Elapsed time: 2786.17s]\n",
      "[Epoch 189/200] [D loss: 0.279450] [G loss: 1.915476] [Elapsed time: 2800.64s]\n",
      "[Epoch 190/200] [D loss: 0.243840] [G loss: 2.639207] [Elapsed time: 2815.62s]\n",
      "[Epoch 191/200] [D loss: 0.237666] [G loss: 2.534870] [Elapsed time: 2830.14s]\n",
      "[Epoch 192/200] [D loss: 0.293672] [G loss: 2.776765] [Elapsed time: 2845.01s]\n",
      "[Epoch 193/200] [D loss: 0.364492] [G loss: 2.845275] [Elapsed time: 2859.84s]\n",
      "[Epoch 194/200] [D loss: 0.289933] [G loss: 1.826300] [Elapsed time: 2874.33s]\n",
      "[Epoch 195/200] [D loss: 0.334376] [G loss: 2.677049] [Elapsed time: 2888.91s]\n",
      "[Epoch 196/200] [D loss: 0.179412] [G loss: 1.981803] [Elapsed time: 2903.51s]\n",
      "[Epoch 197/200] [D loss: 0.223154] [G loss: 2.266187] [Elapsed time: 2918.11s]\n",
      "[Epoch 198/200] [D loss: 0.219701] [G loss: 2.170232] [Elapsed time: 2932.94s]\n",
      "[Epoch 199/200] [D loss: 0.281188] [G loss: 2.861171] [Elapsed time: 2947.38s]\n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "\n",
    "n_epochs = 200 # 학습의 횟수(epoch) 설정\n",
    "sample_interval = 2000 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정\n",
    "start_time = time.time()\n",
    "\n",
    "for epoch in range(n_epochs):\n",
    "    for i, (imgs, _) in enumerate(dataloader):\n",
    "\n",
    "        # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성\n",
    "        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # 진짜(real): 1\n",
    "        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # 가짜(fake): 0\n",
    "\n",
    "        real_imgs = imgs.cuda()\n",
    "\n",
    "        \"\"\" 생성자(generator)를 학습합니다. \"\"\"\n",
    "        optimizer_G.zero_grad()\n",
    "\n",
    "        # 랜덤 노이즈(noise) 샘플링\n",
    "        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()\n",
    "\n",
    "        # 이미지 생성\n",
    "        generated_imgs = generator(z)\n",
    "\n",
    "        # 생성자(generator)의 손실(loss) 값 계산\n",
    "        g_loss = adversarial_loss(discriminator(generated_imgs), real)\n",
    "\n",
    "        # 생성자(generator) 업데이트\n",
    "        g_loss.backward()\n",
    "        optimizer_G.step()\n",
    "\n",
    "        \"\"\" 판별자(discriminator)를 학습합니다. \"\"\"\n",
    "        optimizer_D.zero_grad()\n",
    "\n",
    "        # 판별자(discriminator)의 손실(loss) 값 계산\n",
    "        real_loss = adversarial_loss(discriminator(real_imgs), real)\n",
    "        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)\n",
    "        d_loss = (real_loss + fake_loss) / 2\n",
    "\n",
    "        # 판별자(discriminator) 업데이트\n",
    "        d_loss.backward()\n",
    "        optimizer_D.step()\n",
    "\n",
    "        done = epoch * len(dataloader) + i\n",
    "        if done % sample_interval == 0:\n",
    "            # 생성된 이미지 중에서 25개만 선택하여 5 X 5 격자 이미지에 출력\n",
    "            save_image(generated_imgs.data[:25], f\"{done}.png\", nrow=5, normalize=True)\n",
    "\n",
    "    # 하나의 epoch이 끝날 때마다 로그(log) 출력\n",
    "    print(f\"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dKhzqw6U8u-H"
   },
   "source": [
    "* 생성된 이미지 예시를 출력합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 169
    },
    "execution": {
     "iopub.execute_input": "2023-04-09T11:14:53.049704Z",
     "iopub.status.busy": "2023-04-09T11:14:53.048644Z",
     "iopub.status.idle": "2023-04-09T11:14:53.303825Z",
     "shell.execute_reply": "2023-04-09T11:14:53.302955Z",
     "shell.execute_reply.started": "2023-04-09T11:14:53.049626Z"
    },
    "id": "FeC3eMGa8vc1",
    "outputId": "49201c7d-9673-4555-e4d5-e6e8d2eb214b"
   },
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "No such file or directory: '92000.png'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/display.py:1032\u001b[0m, in \u001b[0;36mImage._data_and_metadata\u001b[0;34m(self, always_both)\u001b[0m\n\u001b[1;32m   1031\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m-> 1032\u001b[0m     b64_data \u001b[38;5;241m=\u001b[39m \u001b[43mb2a_base64\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdata\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241m.\u001b[39mdecode(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mascii\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m   1033\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n",
      "\u001b[0;31mTypeError\u001b[0m: a bytes-like object is required, not 'str'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/formatters.py:973\u001b[0m, in \u001b[0;36mMimeBundleFormatter.__call__\u001b[0;34m(self, obj, include, exclude)\u001b[0m\n\u001b[1;32m    970\u001b[0m     method \u001b[38;5;241m=\u001b[39m get_real_method(obj, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mprint_method)\n\u001b[1;32m    972\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m method \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m--> 973\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mmethod\u001b[49m\u001b[43m(\u001b[49m\u001b[43minclude\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minclude\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mexclude\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mexclude\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    974\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    975\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/display.py:1022\u001b[0m, in \u001b[0;36mImage._repr_mimebundle_\u001b[0;34m(self, include, exclude)\u001b[0m\n\u001b[1;32m   1020\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39membed:\n\u001b[1;32m   1021\u001b[0m     mimetype \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_mimetype\n\u001b[0;32m-> 1022\u001b[0m     data, metadata \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_data_and_metadata\u001b[49m\u001b[43m(\u001b[49m\u001b[43malways_both\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m)\u001b[49m\n\u001b[1;32m   1023\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m metadata:\n\u001b[1;32m   1024\u001b[0m         metadata \u001b[38;5;241m=\u001b[39m {mimetype: metadata}\n",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/display.py:1034\u001b[0m, in \u001b[0;36mImage._data_and_metadata\u001b[0;34m(self, always_both)\u001b[0m\n\u001b[1;32m   1032\u001b[0m     b64_data \u001b[38;5;241m=\u001b[39m b2a_base64(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdata)\u001b[38;5;241m.\u001b[39mdecode(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mascii\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m   1033\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[0;32m-> 1034\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mFileNotFoundError\u001b[39;00m(\n\u001b[1;32m   1035\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNo such file or directory: \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdata)) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01me\u001b[39;00m\n\u001b[1;32m   1036\u001b[0m md \u001b[38;5;241m=\u001b[39m {}\n\u001b[1;32m   1037\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmetadata:\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: No such file or directory: '92000.png'"
     ]
    },
    {
     "ename": "FileNotFoundError",
     "evalue": "No such file or directory: '92000.png'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/display.py:1032\u001b[0m, in \u001b[0;36mImage._data_and_metadata\u001b[0;34m(self, always_both)\u001b[0m\n\u001b[1;32m   1031\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m-> 1032\u001b[0m     b64_data \u001b[38;5;241m=\u001b[39m \u001b[43mb2a_base64\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdata\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241m.\u001b[39mdecode(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mascii\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m   1033\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n",
      "\u001b[0;31mTypeError\u001b[0m: a bytes-like object is required, not 'str'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/formatters.py:343\u001b[0m, in \u001b[0;36mBaseFormatter.__call__\u001b[0;34m(self, obj)\u001b[0m\n\u001b[1;32m    341\u001b[0m     method \u001b[38;5;241m=\u001b[39m get_real_method(obj, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mprint_method)\n\u001b[1;32m    342\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m method \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m--> 343\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mmethod\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    344\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    345\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/display.py:1054\u001b[0m, in \u001b[0;36mImage._repr_png_\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1052\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_repr_png_\u001b[39m(\u001b[38;5;28mself\u001b[39m):\n\u001b[1;32m   1053\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39membed \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mformat \u001b[38;5;241m==\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_FMT_PNG:\n\u001b[0;32m-> 1054\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_data_and_metadata\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/usr/local/lib/python3.9/dist-packages/IPython/core/display.py:1034\u001b[0m, in \u001b[0;36mImage._data_and_metadata\u001b[0;34m(self, always_both)\u001b[0m\n\u001b[1;32m   1032\u001b[0m     b64_data \u001b[38;5;241m=\u001b[39m b2a_base64(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdata)\u001b[38;5;241m.\u001b[39mdecode(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mascii\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m   1033\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[0;32m-> 1034\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mFileNotFoundError\u001b[39;00m(\n\u001b[1;32m   1035\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNo such file or directory: \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdata)) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01me\u001b[39;00m\n\u001b[1;32m   1036\u001b[0m md \u001b[38;5;241m=\u001b[39m {}\n\u001b[1;32m   1037\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmetadata:\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: No such file or directory: '92000.png'"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "\n",
    "Image('92000.png')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "논문에서는 k = 1로 두었지만 D의 학습 양을 k > 1로 두고 학습 시키고 싶었습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "\n",
    "n_epochs = 200 # Number of epochs\n",
    "sample_interval = 2000 # Print results every `sample_interval` batches\n",
    "k = 5 # Number of times to update the discriminator per generator update\n",
    "start_time = time.time()\n",
    "\n",
    "for epoch in range(n_epochs):\n",
    "    for i, (imgs, _) in enumerate(dataloader):\n",
    "\n",
    "        # Real and fake labels\n",
    "        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # Real: 1\n",
    "        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # Fake: 0\n",
    "\n",
    "        real_imgs = imgs.cuda()\n",
    "\n",
    "        \"\"\" Update the discriminator k times. \"\"\"\n",
    "        for j in range(k):\n",
    "            optimizer_D.zero_grad()\n",
    "\n",
    "            # Discriminator loss for real images\n",
    "            real_loss = adversarial_loss(discriminator(real_imgs), real)\n",
    "\n",
    "            # Discriminator loss for fake images\n",
    "            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()\n",
    "            generated_imgs = generator(z).detach() # Detach to avoid computing gradients for the generator\n",
    "            fake_loss = adversarial_loss(discriminator(generated_imgs), fake)\n",
    "\n",
    "            # Total discriminator loss\n",
    "            d_loss = (real_loss + fake_loss) / 2\n",
    "\n",
    "            # Update the discriminator\n",
    "            d_loss.backward()\n",
    "            optimizer_D.step()\n",
    "\n",
    "        \"\"\" Update the generator once. \"\"\"\n",
    "        optimizer_G.zero_grad()\n",
    "\n",
    "        # Generator loss\n",
    "        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()\n",
    "        generated_imgs = generator(z)\n",
    "        g_loss = adversarial_loss(discriminator(generated_imgs), real)\n",
    "\n",
    "        # Update the generator\n",
    "        g_loss.backward()\n",
    "        optimizer_G.step()\n",
    "\n",
    "        done = epoch * len(dataloader) + i\n",
    "        if done % sample_interval == 0:\n",
    "            # Save 25 generated images as a 5x5 grid\n",
    "            save_image(generated_imgs.data[:25], f\"{done}.png\", nrow=5, normalize=True)\n",
    "\n",
    "    # Print log at the end of each epoch\n",
    "    print(f\"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]\")\n"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "name": "GAN for MNIST Tutorial",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
