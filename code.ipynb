

import pandas
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torchvision import models
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.nn.modules.loss import _WeightedLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
import torch.utils.data as data

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])


trainset_path = "/content/NUAEasy/Train"
testset_path = "/content/NUAEasy/Test"


trainset = ImageFolder(trainset_path, transform=transform_train)
testset = ImageFolder(testset_path, transform=transform_test)


print(len(trainset))
val_size = int (0.2*len(trainset))
print(val_size)
train_size = (len(trainset)-val_size)
print(train_size)

train_set,val_set = torch.utils.data.random_split(trainset, [train_size,val_size])

batch_size = 16
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('real', 'spoof')

dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images.shape)

class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=1, stride=1),                         #[B,3,256,256]  ->  [B,64,256,256]
        nn.BatchNorm2d(64),
        nn.ReLU()
        )
        self.low  = nn.Sequential(
        nn.Conv2d(64,128,kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128,204,kernel_size=1, stride=1),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(204,128,kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2)
        )
        self.mid  = nn.Sequential(
        nn.Conv2d(128,153,kernel_size=1, stride=1),
        nn.BatchNorm2d(153),
        nn.ReLU(),
        nn.Conv2d(153,128,kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128,179,kernel_size=1, stride=1),
        nn.BatchNorm2d(179),
        nn.ReLU(),
        nn.Conv2d(179,128,kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2)
        )
        self.high  = nn.Sequential(
        nn.Conv2d(128,153,kernel_size=1, stride=1),
        nn.BatchNorm2d(153),
        nn.ReLU(),
        nn.Conv2d(153,128,kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(384,128,kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128,1,kernel_size=1, stride=1),
        nn.BatchNorm2d(1),
        nn.ReLU()
        )
        self.downsample32X32 = nn.Upsample((32,32), mode='bilinear')

        self.fc_model = nn.Sequential(
        nn.Linear(1*32*32,200),
        nn.Tanh(),
        nn.Linear(200,100),
        nn.Tanh(),
        nn.Linear(100,2)
        )


    def forward(self, x):
        x = self.conv1(x)
        low = self.low(x)
        x_low_32X32 = self.downsample32X32(low)
        mid = self.mid(low)
        x_mid_32X32 = self.downsample32X32(mid)
        high = self.high(mid)
        x_high_32X32 = self.downsample32X32(high)
        x_concat = torch.cat((x_low_32X32,x_mid_32X32,x_high_32X32), dim=1)
        x = self.conv2(x_concat)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)

        return x


net = NewModel()


out = net(images)
print(out.shape)


def evaluation(dataloader, model):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total

class SmoothHTERLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)
        print(targets,log_preds)
        outputs = torch.argmax(targets, dim=1)
        labels = torch.argmax(log_preds, dim=1)
        TP = ((outputs == 1) & (labels == 1)).sum().item()
        TN = ((outputs == 0) & (labels == 0)).sum().item()
        FP = ((outputs== 1) & (labels == 0)).sum().item()
        FN = ((outputs== 0) & (labels == 1)).sum().item()
        print(TP,TN)
        print(FP,FN)



        if FP == 0 and FN  == 0:
          FAR = 0
          FRR = 0
        elif FP  == 0:
          FAR = 0
          FRR = FN/(FN+TP)
        else :
          FRR = 0
          FAR = FP/(FP+TN)
        HTER = (FRR+FAR)/2

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))

opt = optim.Adam(net.parameters())
htloss  = SmoothHTERLoss()

%%time
loss_arr = []
loss_epoch_arr = []
max_epochs = 2

for epoch in range(max_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt.zero_grad()
        outputs = net(inputs)
        loss = htloss(outputs, labels)
        loss.backward()
        opt.step()

    loss_arr.append(loss.item())

    loss_epoch_arr.append(loss.item())
    print('Epoch: %d/%d, Test acc: % 0.2f, Train acc: %0.2f' % (epoch, max_epochs ,evaluation(valloader, net), evaluation(trainloader, net)))
plt.plot(loss_epoch_arr)
plt.show()



def evaluation(dataloader, model):
    total, correct = 0, 0
    true_positive, true_negative = 0, 0
    false_positive, false_negative = 0, 0

    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

        true_positive += ((pred == 1) & (labels == 1)).sum().item()
        true_negative += ((pred == 0) & (labels == 0)).sum().item()
        false_positive += ((pred == 1) & (labels == 0)).sum().item()
        false_negative += ((pred == 0) & (labels == 1)).sum().item()

    return {
        'accuracy': 100 * correct / total,
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative
    }

evaluation_result = evaluation(testloader, net)

true_positive = evaluation_result['true_positive']
true_negative = evaluation_result['true_negative']
false_positive = evaluation_result['false_positive']
false_negative = evaluation_result['false_negative']

false_acceptance_rate = false_positive / (false_positive + true_negative)
false_rejection_rate = false_negative / (false_negative + true_positive)
half_total_error_rate = (false_acceptance_rate + false_rejection_rate) / 2

print("True Positives:", true_positive)
print("True Negatives:", true_negative)
print("False Positives:", false_positive)
print("False Negatives:", false_negative)
print("False Acceptance Rate (FAR):",false_acceptance_rate)
print("False Rejection Rate (FRR):", false_rejection_rate)
print("Half Total Error Rate (HTER):", half_total_error_rate)
