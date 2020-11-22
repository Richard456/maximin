import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_utils import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import *
import numpy as np

dataset_name = 'mnist_m'
image_root = os.path.join('dataset',dataset_name)
model_root = os.path.join('saved_models')
os.makedirs(model_root, exist_ok=True)

cudnn.benchmark = True
lr = 3e-4
batch_size = 256
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

train_list = os.path.join(image_root, 'mnist_m_train_labels.txt')

train_dataset = GetLoader(
    data_root=os.path.join(image_root, 'mnist_m_train'),
    data_list=train_list,
    transform=img_transform
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)


test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

test_dataset = GetLoader(
    data_root=os.path.join(image_root, 'mnist_m_test'),
    data_list=test_list,
    transform=img_transform
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
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

def train_one_epoch(model, dataloader, epoch):
    model.train()

    for i, (img, label) in enumerate(dataloader):
        # training model using source data
        img = img.cuda()
        label = label.cuda()
        output = model(input_data=img)
        loss = loss_func(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i+1, loss))

def test(model, dataloader, epoch):

    """ training """
    model = model.eval()
    model = model.cuda()

    # i = 0
    n_total = 0
    n_correct = 0

    for img, label in dataloader:
        batch_size = img.shape[0]
        img = img.cuda()
        label = label.cuda()

        output = model(input_data=img)
        pred = output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('Epoch: {}, Test Acc: {}'.format(epoch, accu))

# training
for epoch in range(n_epoch):

    train_one_epoch(model, train_dataloader, epoch)
    test(model, test_dataloader, epoch)

torch.save(model, '{0}/mnistm_model.pt'.format(model_root))
