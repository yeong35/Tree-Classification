from __future__ import print_function

import sys
sys.path.append("../")

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset, TreeDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F

from multiprocessing import freeze_support

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

if __name__=='__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=33, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('-train', action='store_true', help="model train")

    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='trainval')

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'tree':
        train_dataset = TreeDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='train')

        valid_dataset = TreeDataset(
            root=opt.dataset,
            split='valid',
            npoints=opt.num_points,
            data_augmentation=False)
        
        test_dataset = TreeDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

    print("train:", len(train_dataloader), "valid:", len(valid_dataloader), "test:", len(test_dataloader))

    num_classes = len(train_dataset.classes)
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))


    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    num_batch = len(train_dataset) / opt.batchSize

writer = SummaryWriter()
if opt.train:
    for epoch in range(opt.nepoch):
        scheduler.step()
        for i, data in enumerate(train_dataloader, 0):

            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

        writer.add_scalar("loss/train", loss.item(), epoch)
        writer.add_scalar("accuracy/train", correct.item() / float(opt.batchSize), epoch)

        print('[%d] train loss: %f accuracy: %f AUC: %f' % (epoch, loss.item(), correct.item() / float(opt.batchSize), roc_auc_score(target.detach().cpu(), pred_choice.detach().cpu())))


        valid_correct = 0
        valid_testset = 0

        for i,data in enumerate(valid_dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            valid_correct += correct.item()
            valid_testset += points.size()[0]

        writer.add_scalar("loss/valid", loss.item(), epoch)
        writer.add_scalar("accuracy/valid", valid_correct / float(valid_testset), epoch)
        
        print('[%d] valid loss: %f accuracy: %f AUC: %f' % (epoch, loss.item(), valid_correct / float(valid_testset), roc_auc_score(target.detach().cpu(), pred_choice.detach().cpu())))

        if best_loss > loss:
            best_loss = loss
            torch.save(classifier.state_dict(), '%s/best_cls_model.pth' % (opt.outf))

else:
    total_correct = 0
    total_testset = 0

    pred_list = []
    target_list = []

    classifier.load_state_dict(torch.load(opt.outf+'/best_cls_model.pth'))

    for i,data in enumerate(test_dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        pred_list += pred_choice.tolist()
        target_list += target.tolist()

        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
    print('AUC {}'.format(roc_auc_score(target.detach().cpu(), pred_choice.detach().cpu())))

print("Done")