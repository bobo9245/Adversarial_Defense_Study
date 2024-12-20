{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#기본 IMPORT\n",
    "import numpy as np\n",
    "from matplotlib import cm\n",
    "import matplotlib.pyplot as plt\n",
    "import math\n",
    "import os\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "\n",
    "from torchvision import transforms, datasets\n",
    "from torchvision.utils import save_image\n",
    "from torch.autograd import Variable\n",
    "\n",
    "from sklearn.metrics import f1_score, confusion_matrix\n",
    "from sklearn.manifold import TSNE\n",
    "\n",
    "#TensorBoard 사용하기\n",
    "from torch.utils.tensorboard import SummaryWriter\n",
    "writer = SummaryWriter()\n",
    "\n",
    "# 출력 줄수 제한 없애기\n",
    "from IPython.core.interactiveshell import InteractiveShell\n",
    "InteractiveShell.ast_node_interactivity=\"all\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 5\n",
    "\n",
    "train_dataset = datasets.MNIST(root='data/MNIST', train=True, download=True, transform=transforms.ToTensor())\n",
    "test_dataset = datasets.MNIST(root='data/MNIST', train=False, transform=transforms.ToTensor())\n",
    "train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
    "test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
    "\n",
    "# Teacher model: pretrained ResNet\n",
    "teacher_model = models.resnet50(pretrained=True)\n",
    "teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)\n",
    "teacher_model = teacher_model.to(DEVICE).eval()\n",
    "\n",
    "# Student model with convolutional layers\n",
    "class ConvStudentNet(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(ConvStudentNet, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)\n",
    "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)\n",
    "        self.fc1 = nn.Linear(64 * 7 * 7, 128)\n",
    "        self.fc2 = nn.Linear(128, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = F.relu(self.conv1(x))\n",
    "        x = F.max_pool2d(x, 2)\n",
    "        x = F.relu(self.conv2(x))\n",
    "        x = F.max_pool2d(x, 2)\n",
    "        x = x.view(-1, 64 * 7 * 7)\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = self.fc2(x)\n",
    "        return x\n",
    "\n",
    "student_model = ConvStudentNet().to(DEVICE)\n",
    "\n",
    "T = 5.0\n",
    "alpha = 0.7\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Defensive Distillation training\n",
    "for epoch in range(EPOCHS):\n",
    "    student_model.train()\n",
    "    for images, labels in train_loader:\n",
    "        images, labels = images.to(DEVICE), labels.to(DEVICE)\n",
    "        with torch.no_grad():\n",
    "            teacher_logits = teacher_model(images)\n",
    "            soft_labels = F.softmax(teacher_logits / T, dim=1)\n",
    "        optimizer.zero_grad()\n",
    "        student_logits = student_model(images)\n",
    "        loss_ce = criterion(student_logits, labels)\n",
    "        loss_distill = F.kl_div(F.log_softmax(student_logits / T, dim=1), soft_labels, reduction='batchmean') * (T ** 2)\n",
    "        loss = alpha * loss_ce + (1 - alpha) * loss_distill\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "print(\"Defensive distillation complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py310",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
