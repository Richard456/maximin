##############################################################
# Generate adversarial datapoints
import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_utils import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import MNISTModel
import numpy as np
from pgd_attack import LinfPGDAttack

dataset_name = 'MNIST'
image_root = os.path.join('dataset', dataset_name)
model_path = os.path.join('saved_models', 'mnist_model.pt')
cudnn.benchmark = True
batch_size = 200
image_size = 28

# manual_seed = random.randint(1, 10000)
manual_seed = 1
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
img_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(
    root='dataset',
    train=True,
    transform=img_transform,
    download=True
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

test_dataset = datasets.MNIST(
    root='dataset',
    train=False,
    transform=img_transform,
    download=True
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8)

# load model
model = torch.load(model_path)

# setup optimizer
loss_func = torch.nn.CrossEntropyLoss()

model = model.cuda()
loss_func = loss_func.cuda()

model = model.eval()
model = model.cuda()

attacker = LinfPGDAttack(model, eps=0.3, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=True)

n_total = 0
n_correct = 0

train_adv_data = []
train_adv_labels = []

for i, (img, label) in enumerate(train_dataloader):
    #img = img.expand(img.data.shape[0], 3, 28, 28)
    
    batch_size = img.shape[0]
    img = img.cuda()
    label = label.cuda()

    adv_img = attacker.perturb(img, label)
    train_adv_data.extend(adv_img.cpu().numpy())
    train_adv_labels.extend(label.cpu().numpy())

    adv_output= model(input_data=adv_img)
    pred = adv_output.data.max(1, keepdim=True)[1]
    n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    n_total += batch_size
    print('Process {}'.format(n_total))

accu = n_correct.data.numpy() * 1.0 / n_total

print('Adv acc:', accu)

adv_data_save_path_train = 'dataset/adv_mnist/train'
os.makedirs(adv_data_save_path_train, exist_ok=True)

np.save(adv_data_save_path_train, [train_adv_data, train_adv_labels])

n_total = 0
n_correct = 0

test_adv_data = []
test_adv_labels = []

for i, (img, label) in enumerate(test_dataloader):
    #img = img.expand(img.data.shape[0], 3, 28, 28)
    
    batch_size = img.shape[0]
    img = img.cuda()
    label = label.cuda()
    adv_img = attacker.perturb(img, label)
    test_adv_data.extend(adv_img.cpu().numpy())
    test_adv_labels.extend(label.cpu().numpy())

    adv_output= model(input_data=adv_img)
    pred = adv_output.data.max(1, keepdim=True)[1]
    n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    n_total += batch_size
    print('Process {}'.format(n_total))

accu = n_correct.data.numpy() * 1.0 / n_total

print('Adv acc:', accu)

adv_data_save_path_test = 'dataset/adv_mnist/test'
os.makedirs(adv_data_save_path_test, exist_ok=True)

np.save(adv_data_save_path_test, [test_adv_data, test_adv_labels])

