
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import torch.backends.cudnn as cudnn

import numpy as np
import torchvision.transforms as transforms


import os
import argparse
import sys
from  sklearn.linear_model import LinearRegression

sys.path.append("../..")
import backbones.cifar as models
from datasets import CellMT
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
parser.add_argument('--bs', default=10, type=int, help='batch size')
parser.add_argument('--es', default=100, type=int, help='epoch size')

parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate without training')




from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.labels_idx)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets,
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
    freeze_backbone=True
    # checkpoint
    # args.checkpoint = './checkpoints/MultiTask/{}'.format(args.arch)
    # if not os.path.isdir(args.checkpoint):
    #     mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([

        transforms.RandomCrop(896),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),

        transforms.ToTensor(),



    ])

    transform_test = transforms.Compose([

        transforms.CenterCrop(896),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),

        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])
    '''13 cell lines'''
    full_data_path = '../../data/'




    val_size = 0.2

    #label_id_path_file= None
    # 44 cell lines
    label_id_path_file = "./train_class_idx.txt"

    if not args.evaluate:

        full_trainset = CellMT(train_data_path=full_data_path
                               ,train=True,transform=transform_train,label_id_path_file=label_id_path_file)

        train_dataset = full_trainset
        validation_dataset = full_trainset

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.bs, shuffle=False, num_workers=2)

        train_class_num = full_trainset.class_number


    #else:
        testset = CellMT(train_data_path=test_data_path,
                              train=False,transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

        train_class_num = testset.class_number

    args.checkpoint = './checkpoints/MultiTask_Test_Data/{}_val{}_transfer_learning'.format(args.arch,val_size)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    # Model
    print('==> Building model..')
    # Sigmoid limit the output time range to (-1,1)


    if args.arch == 'Xception':

        net = models.xception_cla(num_class=44,input_channel=1)

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

            '''if just resume training, unannotate this'''
            #start_epoch = checkpoint['epoch']
            #logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)

            logger = Logger(os.path.join(args.checkpoint, 'log.txt'))

            logger.set_names(['Epoch', 'Learning Rate', 'Train Cla Loss', 'Train Cla Acc', 'Train MSE Loss'
                                 , 'Val Cla Loss', 'Val Cla Acc', 'Val MSE Loss'])

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        # logger.set_names(
        #     ['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc.', 'Test Loss', 'Test Acc.'])

        logger.set_names(['Epoch', 'Learning Rate','Train Cla Loss','Train Cla Acc' ,'Train MSE Loss'
                             ,'Val Cla Loss','Val Cla Acc','Val MSE Loss'])

    # load orignal model then remove FC layers
    if freeze_backbone:

        if args.arch == 'Xception':
            final_layer = list(net.module.children())
            #print(list(net.module.children()))
            for idx,layer in enumerate(net.module.children()):
                if idx is not len(final_layer)-1:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    print('unfreeze layer {}'.format(final_layer[-1]))


            net.module.fc = nn.Sequential(
                nn.Linear(2048, 512, bias=True),

                nn.Linear(512, 64, bias=True),

                nn.Linear(64, 8, bias=True),

                nn.Linear(8, 1, bias=True),
                nn.ReLU(),

            )
            print(net.module)


    torch.lstsq()

    net.cuda()
    # torchvision models were trained usingnn.CrossEntropyLoss which consists of nn.LogSoftmax and then nn.NLLLoss.

    optimizer = optim.SGD(net.parameters()
                          , lr=args.lr, momentum=0.9, weight_decay=5e-4)

    #optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # not use balanced loss


    criterion = nn.MSELoss()

    epoch=0
    val_loss_pre = 9999999
    if not args.evaluate:

        for epoch in range(start_epoch, start_epoch + args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))
            adjust_learning_rate(optimizer, epoch, args.lr,step=20)
            train_reg_loss = \
                train(net,trainloader,optimizer,criterion,device)
            # Validation

            val_cla_loss, val_cla_acc, val_reg_loss =0,0,0
            if (epoch+1)%5==0:
                val_reg_loss = Validation(net, testloader, optimizer, criterion, device)
                print('\n Val CLA loss: {} | Val Accuracy: {} | Val MSE Loss: {}'.format(val_cla_loss,val_cla_acc,val_reg_loss))
                if (val_cla_loss+val_reg_loss) < val_loss_pre:
                    print('-------val loss decrease, save best model now-----------')
                    val_loss_pre = (val_cla_loss+val_reg_loss)
                    save_model(net, None, epoch, os.path.join(args.checkpoint,'best_model.pth'))



            logger.append([epoch+1, optimizer.param_groups[0]['lr'], 0,0,train_reg_loss,val_cla_loss,val_cla_acc,val_reg_loss])
            if (epoch+1)%10 == 0:
                save_model(net, None, epoch, os.path.join(args.checkpoint, 'last_model.pth'))
                print('------------Training Finished-----------------!')


    else:

        test(epoch, net, trainloader, testloader, criterion, device,testset)
    logger.close()


# Training
def train(net,trainloader,optimizer,criterion,device):
    net.train()
    train_cla_loss,train_reg_loss = 0,0
    correct = 0
    total = 0
    train_loss = 0

    for batch_idx, (inputs, (class_labels,time_points)) in enumerate(trainloader):
        inputs, class_labels,time_points = inputs.to(device), class_labels.to(device),time_points.to(device)


        optimizer.zero_grad()
        outputs = net(inputs)

        targets = time_points.type(outputs.dtype)
        loss = criterion(outputs.float(), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f'
                     % (train_loss / (batch_idx + 1)))
    return train_loss/(batch_idx+1)




#Validation
def Validation(net,valloader,optimizer,criterion,device):
    net.eval()


    val_cla_loss, val_reg_loss = 0, 0
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, (class_labels, time_points)) in enumerate(valloader):
            inputs, class_labels, time_points = inputs.to(device), class_labels.to(device), time_points.to(device)




            outputs = net(inputs)

            loss = criterion(outputs.float(), time_points)
            val_loss += loss.item()
            # _, predicted = outputs.max(1)

        return val_loss / (batch_idx + 1)

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

