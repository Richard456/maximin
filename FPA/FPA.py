##############################################################
# Fix point attack for inhomogeneous case
import random
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_utils import ADVMNISTLoader
from torchvision import datasets
from model import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data_utils import GetLoader
from torchvision import transforms
import numpy as np
from dann_pgd_attack import LinfPGDAttack
import imageio
from skimage import img_as_ubyte
from PIL import Image


def defender(loop, args):
    source_dataset_name = 'MNIST'
    target_dataset_name = 'dann_adv_mnistm_recur'
    if loop==0:
        target_dataset_name='adv_mnistm'
    source_image_root = os.path.join('dataset')
    target_image_root = os.path.join('dataset', target_dataset_name)
    model_root = os.path.join('saved_models')
    cudnn.benchmark = True
    lr = 3e-4
    batch_size = 128
    n_epoch = 100

    # Use Random seed to achieve privacy
    # manual_seed = random.randint(1, 10000)
    manual_seed = 1
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load data
    def data_load(eps):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset_source = datasets.MNIST(
            root=source_image_root,
            train=True,
            transform=img_transform,
            download=True
        )

        train_dataloader_source = torch.utils.data.DataLoader(
            dataset=train_dataset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)

        train_dataset_target = ADVMNISTLoader(
            data_path=os.path.join(target_image_root, 'train_eps{}.npy'.format(eps)),
            transform=img_transform
        )

        train_dataloader_target = torch.utils.data.DataLoader(
            dataset=train_dataset_target,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)

        test_dataset_source = datasets.MNIST(
            root=source_image_root,
            train=False,
            transform=img_transform,
            download=True
        )

        test_dataloader_source = torch.utils.data.DataLoader(
            dataset=test_dataset_source,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8)

        test_dataset_target = ADVMNISTLoader(
            data_path=os.path.join(target_image_root, 'test_eps{}.npy'.format(eps)),
            transform=img_transform
        )

        test_dataloader_target = torch.utils.data.DataLoader(
            dataset=test_dataset_target,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8)

        return train_dataloader_source, train_dataloader_target, test_dataloader_source, test_dataloader_target


    # load model
    model = DANNModel()

    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    model = model.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()


    def train_one_epoch(model, dataloader_source, dataloader_target, epoch):
        model.train()

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader:

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            s_img, s_label = data_source_iter.next()
            s_img = s_img.expand(s_img.data.shape[0], 3, 28, 28)

            s_batch_size = s_img.shape[0]
            s_domain_label = torch.zeros(s_batch_size)
            s_domain_label = s_domain_label.long()

            s_img = s_img.cuda()
            s_label = s_label.cuda()
            s_domain_label = s_domain_label.cuda()

            # training model using target data
            t_img, _ = data_target_iter.next()
            t_img = t_img.expand(t_img.data.shape[0], 3, 28, 28)

            t_batch_size = t_img.shape[0]
            t_domain_label = torch.ones(t_batch_size)
            t_domain_label = t_domain_label.long()

            t_img = t_img.cuda()
            t_domain_label = t_domain_label.cuda()

            cat_img = torch.cat((s_img, t_img), 0)
            class_output, domain_output = model(input_data=cat_img, alpha=alpha)
            # s_class_output, s_domain_output = model(input_data=s_img, alpha=alpha)

            s_class_output = class_output[:s_batch_size]
            s_domain_output = domain_output[:s_batch_size]
            t_domain_output = domain_output[s_batch_size:]

            err_s_label = loss_class(s_class_output, s_label)
            err_s_domain = loss_domain(s_domain_output, s_domain_label)
            # _, t_domain_output = model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(t_domain_output, t_domain_label)

            err = err_t_domain + err_s_domain + err_s_label

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            i += 1

            if i % 100 == 0:
                print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                      % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                         err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))


    def test(model, dataloader, dataset_name, epoch):
        alpha = 0

        """ training """
        model = model.eval()
        model = model.cuda()

        # i = 0
        n_total = 0
        n_correct = 0

        for t_img, t_label in dataloader:
            batch_size = t_img.shape[0]
            t_img = t_img.expand(t_img.data.shape[0], 3, 28, 28)
            t_img = t_img.cuda()
            t_label = t_label.cuda()

            class_output, _ = model(input_data=t_img, alpha=alpha)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size

        accu = n_correct.data.numpy() * 1.0 / n_total

        print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))

    #----------------------------------------------------------------------------
    train_dataloader_source, train_dataloader_target, \
    test_dataloader_source, test_dataloader_target = data_load(eps=args.eps)

    # training
    for epoch in range(n_epoch):
        train_one_epoch(model, train_dataloader_source, train_dataloader_target, epoch)
        test(model, test_dataloader_source, source_dataset_name, epoch)
        test(model, test_dataloader_target, target_dataset_name, epoch)

    torch.save(model, '{0}/FPA[{1}]_eps{2}.pth'.format(model_root,loop+1, args.eps))


#------------------------------------------data generation------------------------
def attacker(loop, args):
    if loop==0:
        return
    dataset_name = 'mnist_m'
    image_root = os.path.join('dataset', dataset_name)
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
    model_path = os.path.join('saved_models', 'FPA[{0}]_eps{1}.pth'.format(loop,args.eps))
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

            adv_output,_= model(input_data=adv_img,alpha=0)
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

            adv_output,_= model(input_data=adv_img,alpha=0)
            pred = adv_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
            print('Process {}'.format(n_total))

        accu = n_correct.data.numpy() * 1.0 / n_total

        print('Adv acc:', accu)

        return test_adv_data, test_adv_labels

    #-------------------------------------------------------------------------------------
    attacker = LinfPGDAttack(model, eps=args.eps, nb_iter=100,
                             eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                             targeted=False, num_classes=10, elementwise_best=True)

    adv_data_save_path = os.path.join('dataset','dann_adv_mnistm_recur')
    os.makedirs(adv_data_save_path, exist_ok=True)

    ########### generating train
    train_adv_data, train_adv_labels = train()

    np.save(adv_data_save_path + '/train_eps' + str(args.eps), [train_adv_data, train_adv_labels])

    ########### generating test
    test_adv_data, test_adv_labels = test()

    np.save(adv_data_save_path + '/test_eps' + str(args.eps), [test_adv_data, test_adv_labels])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--eps', default=8 / 255, type=float, help='eps')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(args)

    for i in range(20):
        attacker(i,args)
        defender(i,args)
        print('FPA[{0}] finished.'.format(i))




