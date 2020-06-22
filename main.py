import os
import sys
from itertools import combinations
from distutils.util import strtobool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gensim.models import KeyedVectors
from tqdm import tqdm, trange

from hypergraph import HyperGraph
from utils import load_data, embed_nodes, plot_values
from dataset import QueryDataset
from model import SimpleNN


LOAD_PRETRAINED = False
RESULT_DIR = './result'


def train(model, train_set, val_set, criterion, optimizer, max_epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    train_loader = DataLoader(train_set, batch_size=32, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2)

    loss_log = tqdm(total=0, bar_format='{desc}', position=3)
    acc_log = tqdm(total=0, bar_format='{desc}', position=4)

    for epoch in trange(max_epochs, desc="Epoch", position=2):
        train_loss, train_acc = [], []
        model.train()
        for queries, labels in tqdm(train_loader, desc="Training Iteration", position=1):
            optimizer.zero_grad()
            scores = model(queries)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(scores.data, dim=1)
            acc = (pred == labels).sum().item() / len(labels)

            train_loss.append(loss.item())
            train_acc.append(acc)
            
            des1 = 'Training Loss: {:06.4f}'.format(loss.cpu())
            des2 = 'Training Acc: {:.0%}'.format(acc)
            loss_log.set_description_str(des1)
            acc_log.set_description_str(des2)
            del loss
    
        train_losses.append(sum(train_loss) / len(train_loss))
        train_accuracies.append(sum(train_acc) / len(train_acc))

        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for queries, labels in tqdm(val_loader, desc="Validation Iteration", position=1):
                scores = model(queries)
                loss = criterion(scores, labels)

                _, pred = torch.max(scores.data, dim=1)
                acc = (pred == labels).sum().item() / len(labels)

                val_loss.append(loss.item())
                val_acc.append(acc)

                des1 = 'Validation Loss: {:06.4f}'.format(loss.cpu())
                des2 = 'Validation Acc: {:.0%}'.format(acc)
                loss_log.set_description_str(des1)
                acc_log.set_description_str(des2)
                del loss
        
        val_losses.append(sum(val_loss) / len(val_loss))
        val_accuracies.append(sum(val_acc) / len(val_acc))

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    graph_data = './project_data/paper_author.txt'
    query_data = './project_data/query_public.txt'
    label_data = './project_data/answer_public.txt'
    test_data = './project_data/query_private.txt'

    # Hyperparameters
    learning_rate = 1e-3
    num_epochs = 200
    p, q, p1, q1 = 1, 1, 1, 1
    setting = "hypernode2vec_p("+str(p)+")q("+str(q)+")_p1("+str(p1)+")p2("+str(q1)+")"
    # setting = "node2vec_p("+str(p)+")q("+str(q)+")"
    pretrained_kv_path = "./kvs/"+setting+".kv"
    pretrained_model_path = "./models/"+setting+".pth"

    # Load pretrained vectors, otherwise create a new graph
    if os.path.exists(pretrained_kv_path):
        print("Loading pretrained keyed vectors...")
        node_vectors = KeyedVectors.load(pretrained_kv_path, mmap='r')
    else:
        # TODO: check if we want to create hypernode2vec or node2vec graph
        print("Creating node2vec graph...")
        node_vectors = embed_nodes(graph_data, p=p, q=q)

    # Split and Shuffle Data
    query_train, query_val, label_train, label_val = load_data(query_data, label_data, node_vectors)
    train_set = QueryDataset(query_train, label_train)
    val_set = QueryDataset(query_val, label_val)

    # Load pretrained model, otherwise train parameters
    model = SimpleNN()
    if os.path.exists(pretrained_model_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_set, val_set, criterion, optimizer, max_epochs=num_epochs)
        torch.save(model.state_dict(), pretrained_model_path)

        print("Final training loss: {:06.4f}".format(train_losses[-1]))
        print("Final validation loss: {:06.4f}".format(val_losses[-1]))
        print("Final training accuracy: {:06.4f}".format(train_accuracies[-1]))
        print("Final validation accuracy: {:06.4f}".format(val_accuracies[-1]))

        plot_values(train_losses, val_losses, title="Losses", path="./losses/"+setting+"_loss.png")
        plot_values(train_accuracies, val_accuracies, title="Accuracies", path="./accuracies/"+setting+"_acc.png")

    # TODO: Predict test data  ## save to answer_private.txt
    # predict(test_data)
