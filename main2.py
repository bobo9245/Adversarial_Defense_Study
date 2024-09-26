#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


# In[41]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[42]:


BATCH_SIZE=32
EPOCHS=10

train_dataset = datasets.MNIST(root='data/MNIST',train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='data/MNIST',train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=False)


# In[43]:


for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(),'type:',X_train.type())
    print('y_train:',y_train.size(), 'type:',y_train.type())


# In[44]:


pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize)) #10개 plot하기 위한 figure 크기 설정

for i in range(10):
    plt.subplot(1, 10, i + 1) # plot.subplot(rows, columns, index)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
    plt.title('Class: ' + str(y_train[i].item()))


# In[45]:


class Net(nn.Module): # nn.Module은 모든 neural network의 base class라고 한다. 
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self,x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


# In[46]:


model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
criterion=nn.CrossEntropyLoss()

print(model)


# In[47]:


# 학습 함수
def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed.")


# In[48]:


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= (len(test_loader.dataset)/ BATCH_SIZE)
    test_accuracy = 100. * correct/len(test_loader.dataset)
    return test_loss, test_accuracy


# In[49]:


import random

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def post_training_visualization(model, data_loader, epsilon, device='cuda'):
    model.eval()  # 모델을 평가 모드로 설정
    images, adv_images, labels, adv_labels = [], [], [], []

    # 데이터 로더를 통해 이미지를 가져옴
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # 원본 이미지의 예측
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # Gradient 계산
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # Adversarial example 생성
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]  # 최종 예측

        # 클래스 변경이 있는 경우에만 저장
        for i in range(data.size(0)):
            if init_pred[i] != final_pred[i]:
                images.append(data[i])
                adv_images.append(perturbed_data[i])
                labels.append(init_pred[i].item())
                adv_labels.append(final_pred[i].item())

        # 이미지가 5개 이상 모이면 중단
        if len(images) >= 5:
            break

    # 시각화
    visualize_adversarial_examples(model, images, adv_images, labels, adv_labels, "Post-Training")

def visualize_adversarial_examples(model, original_images, adversarial_images, original_labels, adversarial_labels, title):
    plt.figure(figsize=(10, 5))
    for i in range(len(original_images)):
        # 원본 이미지
        plt.subplot(2, 5, i + 1)
        plt.title(f"Orig: {original_labels[i]}")
        plt.imshow(original_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')

        # Adversarial 이미지
        plt.subplot(2, 5, i + 6)
        plt.title(f"Adv: {adversarial_labels[i]}")
        plt.imshow(adversarial_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')

    plt.suptitle(title)
    plt.show()


# In[50]:


# 학습 실행
train(model, DEVICE, train_loader, optimizer, criterion, EPOCHS)

# 시각화 실행
post_training_visualization(model, test_loader, epsilon, DEVICE)

