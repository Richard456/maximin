import random
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
import imageio
from skimage import img_as_ubyte
from PIL import Image
import numpy as np
from advertorch.attacks import LinfPGDAttack
import sys
sys.path.append('../')
from data_utils import GetLoader

dataset_name = 'mnist_m'
image_root = os.path.join('../dataset', dataset_name)
cudnn.benchmark = True
batch_size = 200
image_size = 28

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
model_path = os.path.join('saved_models', 'PGD_advT_model.pth')
model = torch.load(model_path)

# setup optimizer
loss_func = torch.nn.CrossEntropyLoss()

model = model.cuda()
loss_func = loss_func.cuda()

model = model.eval()
model = model.cuda()


def train():
    n_total = 0
    n_correct = 0

    train_adv_data = []
    train_adv_labels = []

    for i, (img, label) in enumerate(train_dataloader):
        batch_size = img.shape[0]
        img = img.cuda()
        label = label.cuda()
        adv_img = attacker.perturb(img, label)
        train_adv_data.extend(adv_img.cpu().numpy())
        train_adv_labels.extend(label.cpu().numpy())

        adv_output = model(input_data=adv_img)
        pred = adv_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        print('Process {}'.format(n_total))

        # per sample checking
        # for idx in range(adv_img.shape[0]):
        #     tosave = adv_img[idx].cpu().numpy()
        #     tosave = np.moveaxis(tosave, 0, -1)
        #     imageio.imwrite(str(idx) + '.png', img_as_ubyte(tosave))

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('Adv acc:', accu)

    return train_adv_data, train_adv_labels


def test():
    n_total = 0
    n_correct = 0

    test_adv_data = []
    test_adv_labels = []

    for i, (img, label) in enumerate(test_dataloader):
        batch_size = img.shape[0]
        img = img.cuda()
        label = label.cuda()
        adv_img = attacker.perturb(img, label)
        test_adv_data.extend(adv_img.cpu().numpy())
        test_adv_labels.extend(label.cpu().numpy())

        adv_output = model(input_data=adv_img)
        pred = adv_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        print('Process {}'.format(n_total))

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('Adv acc:', accu)

    return test_adv_data, test_adv_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--eps', default=8/255, type=float, help='eps')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    attacker = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss().cuda(), eps=args.eps,
                               nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1.0,
                               targeted=False)

    adv_data_save_path = os.path.join('dataset','linf_mnistm_advT')
    os.makedirs(adv_data_save_path, exist_ok=True)

    ########### generating train
    train_adv_data, train_adv_labels = train()

    np.save(adv_data_save_path + '/train_eps' + str(args.eps), [train_adv_data, train_adv_labels])

    ########### generating test
    test_adv_data, test_adv_labels = test()

    np.save(adv_data_save_path + '/test_eps' + str(args.eps), [test_adv_data, test_adv_labels])
