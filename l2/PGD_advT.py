import random
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torchvision import datasets
from torchvision import transforms
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import L2PGDAttack
import sys
sys.path.append('../')
from data_utils import *
from model import *

target_dataset_name = 'mnist_m'
target_image_root = os.path.join('../dataset', target_dataset_name)
model_root = os.path.join('saved_models')
cudnn.benchmark = True
lr = 3e-4
batch_size = 128
image_size = 28
n_epoch = 100

# manual_seed = random.randint(1, 10000)
manual_seed = 1
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
img_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

train_dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'mnist_m_train'),
    data_list=train_list,
    transform=img_transform
)

train_dataloader_target = torch.utils.data.DataLoader(
    dataset=train_dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

test_list = os.path.join(target_image_root, 'mnist_m_test_labels.txt')

test_dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'mnist_m_test'),
    data_list=test_list,
    transform=img_transform
)

test_dataloader_target = torch.utils.data.DataLoader(
    dataset=test_dataset_target,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8)

# load model
model = MNISTM_Model()

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_func = torch.nn.CrossEntropyLoss()

model = model.cuda()
loss_func = loss_func.cuda()

adversary_train = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss().cuda(), eps=80 / 255,
                               nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1.0,
                               targeted=False)

adversary_test = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss().cuda(), eps=80 / 255,
                               nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1.0,
                               targeted=False)


def train_one_epoch(model, dataloader, epoch):
    model.train()
    n_total = 0
    n_correct = 0
    for i, (img, label) in enumerate(dataloader):
        # training model using source data
        img = img.expand(img.data.shape[0], 3, 28, 28)
        img = img.cuda()
        label = label.cuda()

        with ctx_noparamgrad_and_eval(model):
            img_adv = adversary_train.perturb(img, label)
        output = model(input_data=img_adv)
        loss = loss_func(output, label)
        pred = output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i + 1, loss))
    accu = n_correct.data.numpy() * 1.0 / n_total

    print('Epoch: {}, Train Acc: {}'.format(epoch, accu))
    return accu


def test(model, dataloader, epoch):
    """ training """
    model = model.eval()
    model = model.cuda()

    # i = 0
    n_total = 0
    n_adv_correct = 0
    n_correct = 0

    for img, label in dataloader:
        batch_size = img.shape[0]
        img = img.expand(img.data.shape[0], 3, 28, 28)
        img = img.cuda()
        label = label.cuda()
        with ctx_noparamgrad_and_eval(model):
            img_adv = adversary_test.perturb(img, label)

        adv_output = model(input_data=img_adv)
        pred = adv_output.data.max(1, keepdim=True)[1]
        n_adv_correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        output = model(input_data=img)
        pred = output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('Epoch: {}, Test Acc: {}'.format(epoch, accu))
    accu = n_adv_correct.data.numpy() * 1.0 / n_total

    print('Epoch: {}, Adv Test Acc: {}'.format(epoch, accu))


# training
for epoch in range(n_epoch):
    train_one_epoch(model, train_dataloader, epoch)
    if epoch % 5 == 4:
        test(model, test_dataloader_target, epoch)
    torch.save(model, '{0}/PGD_advT_model.pth'.format(model_root))
