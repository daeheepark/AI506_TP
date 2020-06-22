import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from itertools import combinations
from tensorboardX import SummaryWriter

import copy
import os
from tqdm import tqdm
import numpy as np
import time

from gensim.models import KeyedVectors
from utils_classify import CoauthorshipDataset

DATADIR = './result'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_RATIO = 0.2
EPOCH = 50
LOGING_STEP = 20
SIM_THRES = 0.5


def find_best_thres(ans_list, pred_list, start, end, intv):
    acc_best = 0
    for thres in np.arange(start, end, intv):
        pred_list_th = np.array(pred_list) > thres
        correct = 0
        for gt, pred in zip(ans_list, list(pred_list_th)):
            if gt.item() == pred:
                correct += 1
        acc = correct / len(ans_list)
        if acc_best < acc:
            acc_best = acc
            thres_best = thres
            intv_s, intv_e = thres, thres + intv

    return acc_best, thres_best, intv_s, intv_e

def evaluate(dataloader, network, is_training = False):
    predict_prob = []
    gt_list = []
    network.eval()

    for i, input in tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluating %s Accuracy' % "training" if is_training else 'Evaluating %s Accuracy' % "evaluation"):
        data, label = input
        # data, label = data.to(device), label.to(device)
        output = network(data)

        predict_prob.append(output.item())
        gt_list.append(label[0])

    if is_training == True:
        acc_best, thres_best, intv_s, intv_e = find_best_thres(gt_list, predict_prob, 0, 1, 0.02)
        return acc_best, thres_best
    else:
        pred_list_th = np.array(predict_prob) > SIM_THRES
        correct = 0
        for gt, pred in zip(gt_list, list(pred_list_th)):
            if gt.item() == pred:
                correct += 1
        acc = correct / len(gt_list)
        return acc


class Net(nn.Module):
    def __init__(self, outputdim = 64):
        super(Net, self).__init__()

        self.outputdim = outputdim

        self.fc1 = nn.Linear(128, 96)
        self.fc2 = nn.Linear(96, 72)
        self.fc3 = nn.Linear(72, outputdim)

        self.simfunc = nn.CosineSimilarity()

    def forward(self, x):
        feats = torch.Tensor().reshape(-1, self.outputdim)#.to(device)

        for xi in x.transpose(0,1):
            # xi = xi.unsqueeze(0)
            feat = F.relu(self.fc1(xi))
            feat = F.relu(self.fc2(feat))
            feat = self.fc3(feat)
            feats = torch.cat((feats, feat), 0)

        feats_comb = list(combinations(feats, 2))
        sim = torch.tensor(0., dtype=torch.float)
        for f1, f2 in feats_comb:
            sim = torch.add(sim, self.simfunc(f1.unsqueeze(0), f2.unsqueeze(0)))

        sim = sim/len(feats)
        # sim = torch.sigmoid(sim)

        return sim

def train_and_evaluate(query_fn, label_fn, kv_fn, writer):

    Dataset = CoauthorshipDataset(query_fn, label_fn, kv_fn)
    train_val_length = [Dataset.num_queries - int(VAL_RATIO*Dataset.num_queries), int(VAL_RATIO*Dataset.num_queries)]

    trainset, valset = torch.utils.data.random_split(Dataset, train_val_length)

    trainloader = DataLoader(trainset, shuffle=True)
    valloader = DataLoader(valset, shuffle=False)

    network = Net()#.to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(network.parameters())

    for e in range(EPOCH):
        loss_sum = torch.tensor(0., dtype=torch.float)
        loss = torch.tensor(0., dtype=torch.float)
        for i, input in tqdm(enumerate(trainloader), total=len(trainloader), desc=f'epoch_{e} '):
            data, label = input
            #data, label = data.to(device), label.to(device)
            network.train()
            output = network(data)
            loss_current = loss_fn(output, label)

            if not np.isnan(loss_current.item()): # if only one feature in features, no combination -> zero value -> Nan value

                loss += loss_current
                loss_sum += loss_current

            if i % LOGING_STEP == LOGING_STEP - 1:
                loss.backward()
                optimizer.step()
                loss = torch.tensor(0., dtype=torch.float)

        print('\n', 'loss_sum : ', loss_sum.item())

        acc_tr, SIM_THRES = evaluate(trainloader, network, True)
        print('Training acc : ', acc_tr, "sim thres : ", SIM_THRES, '\n')
        acc_vl = evaluate(valloader, network, False)
        print("Evaluation acc : ", acc_vl, "sim thres : ", SIM_THRES, '\n')
        writer.add_scalar("loss/loss_current", loss_current.item(), e)
        writer.add_scalar("loss/loss_sum", loss_sum.item(), e)
        writer.add_scalar("acc/training_acc", acc_tr, e)
        writer.add_scalar("acc/validation_acc", acc_vl, e)
        writer.add_scalar("optimal threshold", SIM_THRES, e)

    return

QUERY = './project_data/query_public.txt'
LABEL = './project_data/answer_public.txt'
KV_fn = 'hypernode2vec_p(1)q(1)_p1(1)p2(1)dim(128).kv'
KV = os.path.join(DATADIR, KV_fn)
writer = SummaryWriter('runs/'+KV_fn+time.strftime("%Y%m%d_%H:%M:%S"))

if __name__ == "__main__":
    train_and_evaluate(QUERY, LABEL, KV, writer)