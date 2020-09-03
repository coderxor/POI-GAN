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


# 数据加载部分
# define dataset
class POI_Dataset(Dataset):
    def __init__(self, origin_data=None, param=None, train_flag=True):
        self.param = param
        self.data = []
        self.train_flag = train_flag
        if train_flag:
            # data header:user_id,sequence
            for d in origin_data:
                sequence = d['sequence']
                for i in range(len(d['sequence']) - param['other']['sliding_window']):
                    context = d['sequence'][i:i + param['other']['sliding_window']]
                    context_cordi = d['cordis'][i:i + param['other']['sliding_window']]
                    label = d['sequence'][i + param['other']['sliding_window']]
                    label_cordi = d['cordis'][i + param['other']['sliding_window']]
                    tmp = {
                        'user_id': d['user_id'],
                        'context': context,
                        'label': [label],
                        'context_cordi': context_cordi,
                        'label_cordi': [label_cordi]
                    }

                    self.data.append(tmp)
        else:
            all_visits = 0
            for d in origin_data:
                history_check_ins = d['sequence'][:param['other']['sliding_window']]
                target = d['sequence'][param['other']['sliding_window']:]
                history_cordi = d['cordis'][:param['other']['sliding_window']]
                tmp = {
                    'user_id': d['user_id'],
                    'context': history_check_ins,
                    'label': target,
                    'context_cordi': history_cordi,
                    'test_mask': d['test_mask']
                }
                self.data.append(tmp)
                all_visits += len(target)
            print(all_visits)

    def __getitem__(self, index):
        tmp = self.data[index]
        if self.train_flag:
            return torch.LongTensor([tmp['user_id']]), torch.LongTensor(tmp['context']), \
                   torch.zeros(self.param['dataset']['num_pois']).scatter_(0, torch.LongTensor(tmp['label']), 1), \
                   torch.FloatTensor(tmp['context_cordi']), torch.FloatTensor(tmp['label_cordi'])
        else:
            return torch.LongTensor([tmp['user_id']]), torch.LongTensor(tmp['context']), \
                   torch.zeros(self.param['dataset']['num_pois']).scatter_(0, torch.LongTensor(tmp['label']), 1), \
                   torch.FloatTensor(tmp['context_cordi']), \
                   torch.ones(self.param['dataset']['num_pois']).scatter_(0, torch.LongTensor(tmp['test_mask']), 0)

    def __len__(self):
        return len(self.data)


# split check-in sequence by length
def __load_data(path, param):
    split_rate = param['other']['split_rate']
    pois = pd.read_csv(path, sep=' ')
    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_user_cordi = [[i.split(',') for i in upois.split('/')] for upois in pois['u_coordinates']]
    num_users = len(all_user_pois)

    # generate poi2id table
    pois = set()
    for upois in all_user_pois:
        for poi in upois:
            pois.add(poi)
    poi_id = {}
    num_poi = len(pois)
    pid = 0
    for poi in pois:
        poi_id.setdefault(poi, pid)
        pid += 1

    print('users:{v1},pois:{v2}'.format(v1=num_users, v2=num_poi))

    # 将原数据中的poi用id替换
    for i in range(len(all_user_pois)):
        for j in range(len(all_user_pois[i])):
            all_user_pois[i][j] = poi_id[all_user_pois[i][j]]

    # 建立poi坐标表
    poi_id_cordi = {}
    max_x, min_x, max_y, min_y = 0., 2., 100., 104.
    for i in range(len(all_user_pois)):
        for j in range(len(all_user_pois[i])):
            cordi = all_user_cordi[i][j]
            cordi = [float(v) for v in cordi]
            x, y = cordi[0], cordi[1]
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
            poi_id_cordi.setdefault(all_user_pois[i][j], cordi)
    for i in range(len(all_user_cordi)):
        for j in range(len(all_user_cordi[i])):
            all_user_cordi[i][j] = poi_id_cordi[all_user_pois[i][j]]
    all_user_cordi = min_max_normalization(all_user_cordi, max_x, min_x, max_y, min_y)

    # 按序列长度分割数据
    train_data = []
    test_data = []
    u_id = 0
    all_visit = 0
    for i in range(len(all_user_pois)):
        sequence = all_user_pois[i]
        split_point = int(len(sequence) * split_rate)
        train_sequence = sequence[:split_point]
        cordis = all_user_cordi[i]
        train_cordi = cordis[:split_point]
        # 将训练序列的最后sliding_window个作为历史数据用于预测test中的check_in
        test_sequence = sequence[split_point - param['other']['sliding_window']:]
        test_cordi = cordis[split_point - param['other']['sliding_window']:]
        # 测试时如果每个用户只预测一次需要对测试序列去重
        all_visit += len(test_sequence[param['other']['sliding_window']:])
        tmp = []
        for poi in test_sequence[param['other']['sliding_window']:]:
            if poi not in tmp:
                tmp.append(poi)
        test_mask = list(set(train_sequence))
        test_sequence = test_sequence[:param['other']['sliding_window']] + tmp
        train_data.append({'user_id': u_id, 'sequence': train_sequence, 'cordis': train_cordi})
        test_data.append({'user_id': u_id, 'sequence': test_sequence, 'cordis': test_cordi, 'test_mask': test_mask})
        u_id += 1
    print(u_id)
    print(all_visit)
    print(max_x, min_x, max_y, min_y)
    return train_data, test_data, num_users, num_poi


def load_data(param):
    dataset = param['dataset']['name']
    path = './poidata/' + dataset + '/sequence/' + dataset + '.txt'
    return __load_data(path, param)


# 按次序创建目录
def create_experiment_dir(root='./experiment'):
    for _, dirs, _ in os.walk(root):
        if len(dirs):
            for i in range(len(dirs)):
                dirs[i] = int(dirs[i])
            dirs.sort()
            index = dirs[-1] + 1
        else:
            index = 1
        break

    return os.path.join(root, str(index))


# 对坐标值进行归一化min-max
def min_max_normalization(cordis, max_x, min_x, max_y, min_y):
    tmp = []
    for seq in cordis:
        seq_cordi = []
        for cordi in seq:
            x = (cordi[0] - min_x) / (max_x - min_x)
            y = (cordi[1] - min_y) / (max_y - min_y)
            seq_cordi.append([x, y])
        tmp.append(seq_cordi)
    return tmp
