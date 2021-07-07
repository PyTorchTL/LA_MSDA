from __future__ import print_function

import argparse
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
from DataLoader import MSDA_DataLoader as MSDA_DataLoader
from LAMSDA_Modules import mutil_sources_modules as models  ## LA-MSDA(EEGNet+LLMMD+GO)
import numpy as np
from sklearn import metrics
from tensorboardX import SummaryWriter


def train(model, resultList):
    # Optimizer
    params_list = [{'params': model.sharedNet.parameters()}]
    for i in range(opt.num_source):
        params_list.append({'params': model.cls_List_spec[i].parameters(), 'lr': opt.lr})
    for i in range(opt.num_source):
        params_list.append({'params': model.specificNet[i].parameters(), 'lr': opt.lr}, )
    optimizer = torch.optim.SGD(params_list, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.l2_decay)

    # data
    dataS = [[]] * opt.num_source
    labelS = [0] * opt.num_source
    for i in range(opt.num_source):
        dataS[i], labelS[i] = source_loader_list[i].infiniteNext()
    dataT, labelT = target_train_loader.infiniteNext()
    iteration = 0
    correctForNoTrained = 0  # NoTrained of Target data
    correctForTrained = 0  # Trained of Target data
    correctForAll = 0  # All of Target data
    PRF1_for_notrain_narray = 0
    PRF1_for_train_narray = 0
    PRF1_for_all_narray = 0
    isContinue = True

    model.train()
    while isContinue:
        iteration += 1

        if (iteration - 1) % 100 == 0:
            print("[{}][Train][Change Learning Rate]New learning rate：{:.6f}".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), opt.lr))#LEARNING_RATE

        train_loss = 0

        for i in range(opt.num_source):
            optimizer.zero_grad()
            cls_loss, local_loss, glo_loss = model(Variable(dataS[i]), Variable(dataT), Variable(labelS[i]), mark=i,
                                                mmd_type=opt.mmd_type)
            gamma = 2 / (1 + math.exp(-10 * (iteration) / (opt.iteration))) - 1
            mu = 2 / (1 + math.exp(-10 * (iteration) / (opt.iteration))) - 1
            loss = cls_loss + mu * local_loss + gamma * glo_loss

            if iteration % opt.log_interval == 0:
                print(
                    "[{}][Train][Iteration/All(x%): {:.0f}/{:.0f}({:.0f}%)][Source: {:.0f}][Loss: {:.6f}][soft_Loss: {:.6f}][{}: {:.6f}][glo_loss: {:.6f}]".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iteration, opt.iteration,
                        100. * iteration / opt.iteration, i, loss.item(), cls_loss.item(), opt.mmd_type,
                        local_loss.item(),
                        glo_loss.item()))

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        if iteration % (opt.test_interval * 10) == 0:
            train_loss = train_loss / (opt.num_source * len(dataS[0]))


            t_correctForNoTrained, t_loss_correctForNoTrained, t_PRF1_for_notrain_narray = test(model, target_test_loader)
            t_correctForTrained, t_loss_correctForTrained, t_PRF1_for_train_narray = test(model, target_train_loader)

            t_correctForAll = t_correctForNoTrained + t_correctForTrained
            t_PRF1_for_all_narray = t_PRF1_for_notrain_narray + t_PRF1_for_train_narray


            correctForNoTrained = correctForNoTrained if correctForNoTrained > t_correctForNoTrained else t_correctForNoTrained
            correctForTrained = correctForTrained if correctForTrained > t_correctForTrained else t_correctForTrained
            correctForAll = correctForAll if correctForAll > t_correctForAll else t_correctForAll
            PRF1_for_notrain_narray = PRF1_for_notrain_narray if correctForNoTrained > t_correctForNoTrained else t_PRF1_for_notrain_narray
            PRF1_for_train_narray = PRF1_for_train_narray if correctForTrained > t_correctForTrained else t_PRF1_for_train_narray
            PRF1_for_all_narray = PRF1_for_all_narray if correctForAll > t_correctForAll else t_PRF1_for_all_narray

            len_ts = len(target_test_loader.baseloader.dataset)
            len_tt = len(target_train_loader.baseloader.dataset)

            # narry for notrained
            NoTrained_result = {100. * (correctForNoTrained.numpy() / len_ts): t_PRF1_for_notrain_narray} if correctForNoTrained > t_correctForNoTrained else {
                100. * (t_correctForNoTrained.numpy() / len_ts): PRF1_for_notrain_narray}
            # narry for trained
            Trained_result = {100. * (correctForTrained.numpy() / len_tt): PRF1_for_train_narray} if correctForTrained > t_correctForTrained else {
                100. * (t_correctForTrained.numpy() / len_tt): PRF1_for_train_narray}
            # narry for all
            All_result = {100. * (correctForAll.numpy() / (len_ts + len_tt)): (PRF1_for_notrain_narray + PRF1_for_train_narray) / 2} if correctForAll > t_correctForAll else {
                100. * (t_correctForAll.numpy() / (len_ts + len_tt)): (PRF1_for_notrain_narray + PRF1_for_train_narray) / 2}

            print(
                "[{}][Test][Iteration/All(x%): {:.0f}/{:.0f}({:.0f}%)][TrainSubject: {}][TrainLoss: {:.6f}][Max_(NoTrained%/Trained%/All%): {:.2f}%/{:.2f}%/{:.2f}%][Now_(NoTrained%/Trained%/All%): {:.2f}%/{:.2f}%/{:.2f}%]".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iteration, opt.iteration,
                    100. * iteration / opt.iteration, opt.target_name_list, train_loss,
                    100. * correctForNoTrained / len_ts,
                    100. * correctForTrained / len_tt,
                    100. * correctForAll / (len_ts + len_tt),

                    100. * t_correctForNoTrained / len_ts,
                    100. * t_correctForTrained / len_tt,
                    100. * t_correctForAll / (len_ts + len_tt),

                ))


            print('{} \t [the PRF1 of notrained correct] \t {}'.format(opt.target_name_list, NoTrained_result))
            print('{} \t [the PRF1 of trained correct] \t {}'.format(opt.target_name_list, Trained_result))
            print('{} \t [the PRF1 of all correct] \t {}'.format(opt.target_name_list, All_result))

            if iteration == opt.iteration:
                resultList.append(opt.target_name_list)
                resultList.append(NoTrained_result)
                resultList.append(Trained_result)
                resultList.append(All_result)
                resultList.append('\n')
                print('ResultList \n {}'.format(resultList))
            writer.add_scalars('Loss',
                               {'Train_Loss': train_loss, 'Test_loss[No_Trained]': t_loss_correctForNoTrained,
                                'Test_loss[Trained]': t_loss_correctForTrained}, iteration)
            writer.add_scalars('Accuracy',
                               {'NoTrained': 100. * t_correctForNoTrained / len(target_test_loader.baseloader.dataset),
                                'Trained': 100. * t_correctForTrained / len(target_train_loader.baseloader.dataset),
                                'All': 100. * t_correctForAll / (len(target_test_loader.baseloader.dataset) + len(
                                    target_train_loader.baseloader.dataset))}, iteration)


        for i in range(opt.num_source):
            dataS[i], labelS[i] = source_loader_list[i].infiniteNext()
        dataT, labelT = target_train_loader.infiniteNext()
        if iteration > opt.iteration:
            isContinue = False


def test(model, target_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        dataT, labelT = target_loader.getAllData()
        predList = model(Variable(dataT))
        predMean = None
        for one in predList:
            if predMean is None:
                predMean = one
            else:
                predMean += one
        predMean /= opt.num_source
        test_loss += F.nll_loss(F.log_softmax(predMean, dim=1), labelT).item()
        pred_ = predMean.data.max(1)[1]
        correct = pred_.eq(labelT.data.view_as(pred_)).cpu().sum()
        test_loss /= len(target_loader.baseloader.dataset)

        Precision = metrics.precision_score(labelT.cpu().numpy(), pred_.cpu().numpy(), average='binary')
        Recall = metrics.recall_score(labelT.cpu().numpy(), pred_.cpu().numpy(), average='binary')
        F1score = metrics.f1_score(labelT.cpu().numpy(), pred_.cpu().numpy(), average='binary')

    model.training = True

    return correct, test_loss, np.array([100. * Recall, 100. * Precision, 100. * F1score])


if __name__ == '__main__':

    # Training settings
    cuda = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--iteration", type=int, default=500, help="num of the iteration")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--seed", type=int, default=8, help="")
    parser.add_argument("--log_interval", type=int, default=10, help="")
    parser.add_argument("--l2_decay", type=float, default=5e-4, help="")
    parser.add_argument("--class_num", type=int, default=2, help="")   # number of classes
    parser.add_argument("--sample_time", type=float, default=0.5, help="")
    parser.add_argument("--test_rate", type=float, default=0.2, help="Test rate of the dataset")  # auxiliary training data rate
    parser.add_argument("--namelist", default=['data', 'label'], help="")
    parser.add_argument("--shape", default=[61, 27], help="")   # the number of channel is 61, and the frequency points are 27
    parser.add_argument("--num_source", default=14, help="")   #corresponding to 14 source domains
    parser.add_argument("--source_name_list", type=list, default=[], help="")
    parser.add_argument("--target_name_list", type=list, default=[], help="")
    parser.add_argument("--test_interval", type=int, default=1, help="")
    parser.add_argument("--mmd_type", type=str, default='llmmd', help="")
    opt = parser.parse_args()
    print(opt)

    # For cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(opt.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        device = torch.device("cpu")
        torch.manual_seed(opt.seed)
        kwargs = {}

    # load data
    root_path = 'data/'

    domain_list = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
                   's09', 's10', 's11', 's12', 's13', 's14', 's15']

    resultList = []  # Results List

    for _, target in enumerate(domain_list):
        target_name_list = [target]
        source_name_list = []
        for _, source in enumerate(domain_list):
            if target_name_list[0] == source:
                continue
            source_name_list.append(source)

        opt.source_name_list = source_name_list
        opt.target_name_list = target_name_list
        opt.num_source = len(source_name_list)

        # ======== all subjects ========

        source_loader_list = []
        for source_name in source_name_list:
            source_loader_list.append(
                MSDA_DataLoader.load_data(root_path, source_name, opt.batch_size, opt.namelist, opt.shape, 0, device, kwargs))

        target_train_loader = MSDA_DataLoader.load_data(root_path, opt.target_name_list[0], opt.batch_size, opt.namelist,
                                                    opt.shape, opt.test_rate, device,
                                                    kwargs)[0]
        target_test_loader = MSDA_DataLoader.load_data(root_path, opt.target_name_list[0], opt.batch_size, opt.namelist,
                                                   opt.shape, opt.test_rate, device,
                                                   kwargs)[1]

        writer = SummaryWriter(comment='_' + opt.target_name_list[0] + '_' + opt.mmd_type + '_weight')

        # training the model
        model = models.LAMSDA(num_classes=opt.class_num).to(device)
        print(model)
        train(model, resultList)


        # writer.export_scalars_to_json("./result/loss_acc.json")
        writer.close()
