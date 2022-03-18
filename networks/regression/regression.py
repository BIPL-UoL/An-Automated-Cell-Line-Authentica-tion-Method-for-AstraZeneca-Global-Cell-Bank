
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pretrainedmodels
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torchnet
import torchvision.models
import os
import argparse
import sys
from torchvision import models as torch_model
#from models import *
sys.path.append("../..")
import backbones.cifar as models
from datasets import CellRegression
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='Xception', type=str, help='choosing network')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--train_class_num', default=9, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=4, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate without training')




from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.labels_idx)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,

        random_state=10
    )
    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset


args = parser.parse_args()



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # checkpoint


    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([

        transforms.RandomCrop(896),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),


    ])

    transform_test = transforms.Compose([

        transforms.CenterCrop(896),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),

        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])

    '''30 cell lines'''
    full_data_path = '../../DATA_WELL_CV/'
    #full_data_path = '/scratch/depdetect/lt228/Cell_Bank/new_data_split/'

    # train_data_path = full_data_path + 'train'
    # test_data_path = full_data_path + 'val'
    val_size = 0.1
    label_id_path_file = "./train_class_idx.txt"

    if not args.evaluate:

        full_trainset = CellRegression(train_data_path=full_data_path
                               ,train=True,transform=transform_train,label_id_path_file=label_id_path_file,cv_test_fold_id=1)


        #May have imbalanced issue here
        train_dataset, validation_dataset = __balance_val_split(full_trainset,val_split=val_size)

        #full_trainloader = torch.utils.data.DataLoader(full_trainset, batch_size=args.bs, shuffle=True, num_workers=4)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.bs, shuffle=False, num_workers=4)



    else:
        pass
        # testset = CellRegression(train_data_path=test_data_path,
        #                       train=False,transform=transform_test,cv_test_fold_id=1)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

    args.checkpoint = './checkpoints/DATA_WELL_CV/{}_val{}_testFold{}_regression'.format(args.arch, val_size,1)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Model
    print('==> Building model..')
    if args.arch in {'xception','resnet50','resnet18'}:
        net = False
        #net = pretrainedmodels.__dict__[args.arch](num_classes=train_class_num,pretrained=None)

    elif args.arch == 'Xception':
        net = models.xception_cla(num_class=1,input_channel=1)

    else:
        print(args.arch)
        #net = torch_model.__dict__[args.arch](pretrained=False, num_classes=args.train_class_num)
        #net = models.__dict__[args.arch](num_classes=train_class_num) # CIFAR 100
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):

            print('==> Resuming from checkpoint..')
            # change here
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        # logger.set_names(
        #     ['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc.', 'Test Loss', 'Test Acc.'])

        logger.set_names(['Epoch', 'Learning Rate', 'Train MSE Loss','Val MSE Loss','Test Loss'])

    # torchvision models were trained usingnn.CrossEntropyLoss which consists of nn.LogSoftmax and then nn.NLLLoss.
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr,)
    # test(0, net, trainloader, testloader, criterion, device)
    epoch=0
    val_loss_pre = 9999999
    if not args.evaluate:

        for epoch in range(start_epoch, start_epoch + args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))
            adjust_learning_rate(optimizer, epoch, args.lr,step=20)
            train_loss = train(net,trainloader,optimizer,criterion,device)
            # Validation

            test_loss = 0
            val_loss = 0
            if (epoch+1)%5==0:
                val_loss = Validation(net, valloader, optimizer, criterion, device)
                print('\n Validation MES Loss: %f'%(val_loss))
                if val_loss < val_loss_pre:
                    print('-------val loss decrease, save best model now-----------')
                    val_loss_pre = val_loss
                    save_model(net, None, epoch, os.path.join(args.checkpoint,'best_model.pth'))

            #
            # logger.append(
            #     [epoch + 1, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss,
            #      test_acc])

            logger.append([epoch+1, optimizer.param_groups[0]['lr'], train_loss,val_loss, test_loss])
            if (epoch+1)%10 == 0:
                save_model(net, None, epoch, os.path.join(args.checkpoint, 'last_model.pth'))
                print('------------Training Finished-----------------!')


    else:
        test(epoch, net, trainloader, testloader, criterion, device,testset)
    logger.close()


# Training
def train(net,trainloader,optimizer,criterion,device):
    net.train()
    train_loss = 0


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.type(outputs.dtype)
        # print('output {} type: {},target {} type: {} '.
        #       format(outputs,type(outputs),targets,type(targets)))
        loss = criterion(outputs.float(), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f'
            % (train_loss/(batch_idx+1)))
    return train_loss/(batch_idx+1)

#Validation
def Validation(net,valloader,optimizer,criterion,device):
    net.eval()

    val_loss = 0

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # unified float and double

            loss = criterion(outputs.float(), targets)
            val_loss += loss.item()
            # _, predicted = outputs.max(1)



    return val_loss/(batch_idx+1)

def test(epoch, net,trainloader,  testloader,criterion, device,testset):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    #print(net.module)
    predict_results =[]
    scores, labels = [], []
    net_features = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            # _, predicted = outputs.max(1)
            predict_results.append(predicted)
            scores.append(outputs)
            labels.append(targets)
            #net_features.append(net.module.features(inputs))
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader))

    # plot_features = torch.cat(net_features, dim=0).cpu().numpy()
    # print(plot_features.shape)
    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)
    predict_results = np.array(torch.cat(predict_results,dim=0).cpu().numpy())



    print("Evaluation...")

    eval_openmax = Evaluation(predict_results, labels)


    label_names=[]

    with open(testset.label_id_path_file, 'r') as f:
        lines = f.readlines()
        label_names = [line.rstrip() for line in lines]




    print(f"%s accuracy is %.3f,f1_score is %.3f"%(args.arch,eval_openmax.accuracy,eval_openmax.f1_macro))

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import seaborn as sns
    # X_tsne = TSNE(n_components=2,perplexity=50)
    # x_embeded = X_tsne.fit_transform(plot_features)
    # import pandas as pd
    # df_subset = pd.DataFrame()
    # df_subset['tsne-2d-one'] = x_embeded[:, 0]
    # df_subset['tsne-2d-two'] = x_embeded[:, 1]
    # df_subset['Cell Lines'] = [label_names[ids] for ids in labels]
    # plt.figure(figsize=(15, 15))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="Cell Lines",
    #
    #     data=df_subset,
    #     legend="auto",
    # )
    # #plt.savefig('./TSNE.png')
    # plt.show()

    fig,axs=plt.subplots(nrows=1,ncols=1,figsize=(40, 40))
    eval_openmax.plot_confusion_matrix(ax=axs[0],labels=label_names)
    axs[0].set_title('{}'.format(args.arch))

    plt.show()

def save_model(net, acc, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'testacc': acc,
        'epoch': epoch,
    }
    torch.save(state, path)

if __name__ == '__main__':
    main()

