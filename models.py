import os
import random
import time
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from torch import argmax, optim, cosine_similarity
from torch.nn.functional import log_softmax
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


# 模型部分

class Generator(nn.Module):
    def __init__(self, param):
        super(Generator, self).__init__()
        self.param = param
        para = param
        # embeddings of sbj and obj
        self.sbj_embedding = nn.Embedding(num_embeddings=para['dataset']['num_users'],
                                          embedding_dim=para['model']['embedding_dim'], sparse=True)
        self.obj_embedding = nn.Embedding(num_embeddings=para['dataset']['num_pois'],
                                          embedding_dim=para['model']['embedding_dim'], sparse=True)
        self.seq_lstm = nn.LSTM(
            input_size=para['model']['embedding_dim'],
            hidden_size=para['model']['hidden_size'],
            num_layers=para['model']['LSTM_layer'],
            batch_first=True
        )
        self.cordi_lstm = nn.LSTM(
            input_size=2,
            hidden_size=para['model']['hidden_size'],
            num_layers=para['model']['LSTM_layer'],
            batch_first=True
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(
                in_features=2 * para['model']['hidden_size'],
                out_features=para['model']['embedding_dim']
            ),
            nn.Softmax()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(
                in_features=para['model']['embedding_dim'],
                out_features=para['model']['hidden_size'] * 10
            ),
            nn.BatchNorm1d(para['model']['hidden_size'] * 10),
            nn.ReLU(True),
            nn.Linear(
                in_features=para['model']['hidden_size'] * 10,
                out_features=para['dataset']['num_pois']
            ),
            nn.BatchNorm1d(para['dataset']['num_pois']),
            nn.ReLU(True),
            nn.Sigmoid()
        )
        self.cordi_layer = nn.Sequential(
            nn.Linear(
                in_features=para['model']['hidden_size'],
                out_features=para['model']['hidden_size'] // 10
            ),
            nn.BatchNorm1d(para['model']['hidden_size'] // 10),
            nn.ReLU(True),
            nn.Linear(
                in_features=para['model']['hidden_size'] // 10,
                out_features=2
            ),
            nn.BatchNorm1d(2),
            nn.ReLU(True),
            nn.Sigmoid()
        )

    def forward(self, id_input=None, seq_input=None, cordi_input=None):
        para = self.param

        # initialize h_0 and c_0 randomly (i.e., noise)

        h_0 = c_0 = torch.randn(
            (
                para['model']['LSTM_layer'],
                id_input.size()[0],
                para['model']['hidden_size']
            ),
            device=para['other']['device']
        )
        # generate inputs from embedding layer
        lstm_input = self.obj_embedding(seq_input)
        sbj_emb = self.sbj_embedding(id_input).view(-1, para['model']['embedding_dim'])
        # lstm process
        _, (seq_lstm_hidden, _) = self.seq_lstm(lstm_input, (h_0, c_0))
        _, (cordi_lstm_hidden, _) = self.cordi_lstm(cordi_input, (h_0, c_0))
        # generate inputs of attention layer
        attention_input = torch.cat((seq_lstm_hidden[1], cordi_lstm_hidden[1]), 1)

        # calculate attention weights
        attention_weights = self.attention_layer(attention_input.view(-1, 2 * para['model']['hidden_size']))
        # pay attention
        linear_input = torch.mul(attention_weights, sbj_emb)
        # linear layer
        linear_input = linear_input.view(-1, para['model']['embedding_dim'])
        output = self.output_layer(linear_input)
        cordi = self.cordi_layer(cordi_lstm_hidden[1].view(-1, para['model']['hidden_size']))
        return output, cordi


class Discriminator(nn.Module):
    def __init__(self, param):
        super(Discriminator, self).__init__()
        self.param = param
        para = param
        # embeddings of sbj and obj
        self.sbj_embedding = nn.Embedding(num_embeddings=para['dataset']['num_users'],
                                          embedding_dim=para['model']['embedding_dim'], sparse=True)
        self.obj_embedding = nn.Embedding(num_embeddings=para['dataset']['num_pois'],
                                          embedding_dim=para['model']['embedding_dim'], sparse=True)
        self.seq_lstm = nn.LSTM(
            input_size=para['model']['embedding_dim'],
            hidden_size=para['model']['hidden_size'],
            num_layers=para['model']['LSTM_layer'],
            batch_first=True
        )
        self.cordi_lstm = nn.LSTM(
            input_size=2,
            hidden_size=para['model']['hidden_size'],
            num_layers=para['model']['LSTM_layer'],
            batch_first=True
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(
                in_features=2 * para['model']['hidden_size'],
                out_features=para['model']['embedding_dim']
            ),
            nn.Softmax()
        )

        self.recon_layer = nn.Sequential(
            nn.Linear(
                in_features=para['dataset']['num_pois'],
                out_features=para['model']['hidden_size'] * 10
            ),
            nn.BatchNorm1d(para['model']['hidden_size'] * 10),
            nn.ReLU(True),
            nn.Linear(
                in_features=para['model']['hidden_size'] * 10,
                out_features=para['model']['embedding_dim']
            ),
            nn.BatchNorm1d(para['model']['embedding_dim']),
            nn.ReLU(True),
            nn.Sigmoid()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(
                in_features=2 * para['model']['embedding_dim'],
                out_features=2 * para['model']['embedding_dim'] // 10
            ),
            nn.BatchNorm1d(2 * para['model']['embedding_dim'] // 10),
            nn.ReLU(True),
            nn.Linear(
                in_features=2 * para['model']['embedding_dim'] // 10,
                out_features=1
            ),
            nn.BatchNorm1d(1),
            nn.ReLU(True),
            nn.Sigmoid()
        )

    #         self.output_layer = nn.Sequential(
    #             nn.Linear(
    #                 in_features=para['model']['embedding_dim']+para['model']['hidden_size']+para['dataset']['num_pois'],
    #                 out_features=(para['model']['embedding_dim']+para['model']['hidden_size']+para['dataset']['num_pois'])//10
    #             ),
    #             nn.BatchNorm1d((para['model']['embedding_dim']+para['model']['hidden_size']+para['dataset']['num_pois'])//10),
    #             nn.ReLU(True),
    #             nn.Linear(
    #                 in_features=(para['model']['embedding_dim']+para['model']['hidden_size']+para['dataset']['num_pois'])//10,
    #                 out_features=1
    #             ),
    #             nn.BatchNorm1d(1),
    #             nn.ReLU(True),
    #             nn.Sigmoid()
    #         )

    def forward(self, id_input=None, seq_input=None, cordi_input=None, p_vector=None):
        # initialize h_0 and c_0 randomly (i.e., noise)
        para = self.param
        h_0 = c_0 = torch.zeros(
            (
                para['model']['LSTM_layer'],
                id_input.size()[0],
                para['model']['hidden_size']
            ),
            device=para['other']['device']
        )
        # generate inputs from embedding layer
        lstm_input = self.obj_embedding(seq_input)
        sbj_emb = self.sbj_embedding(id_input).view(-1, para['model']['embedding_dim'])
        # lstm process
        _, (seq_lstm_hidden, _) = self.seq_lstm(lstm_input, (h_0, c_0))
        _, (cordi_lstm_hidden, _) = self.cordi_lstm(cordi_input, (h_0, c_0))

        # generate inputs of attention layer
        attention_input = torch.cat((seq_lstm_hidden[1], cordi_lstm_hidden[1]), 1)

        # calculate attention weights
        attention_weights = self.attention_layer(attention_input.view(-1, 2 * para['model']['hidden_size']))
        # pay attention
        attentioned_u_emb = torch.mul(attention_weights, sbj_emb)

        # reconstruct from p_vector
        rec_u_emb = self.recon_layer(p_vector)

        # concatnate the inputs

        linear_input = torch.cat((attentioned_u_emb, rec_u_emb), 1).view(-1, 2 * para['model']['embedding_dim'])
        # linear layer
        output = self.output_layer(linear_input)
        return output