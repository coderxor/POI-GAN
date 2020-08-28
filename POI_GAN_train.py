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
from models import *
from Load_data import *


# 实验部分
class POI_Recommendation_Experiment(object):
    def __init__(self, param):
        self.param = param
        self.load_data()
        self.load_model()

        # 创建实验目录
        if self.param['other']['experiment_name']:
            self.root = os.path.join('./experiment', self.param['other']['experiment_name'])
        if os.path.exists(self.root):
            self.root = create_experiment_dir()
        self.checkpoints_path = os.path.join(self.root, 'checkpoints')
        # self.result_path=os.path.join(self.root,'result')
        os.mkdir(self.root)

        # 打印参数并写入文件
        para_file = os.path.join(self.root, 'parameters.txt')
        f = open(para_file, 'w')
        for i in self.param:
            s = i + ':'
            print(s)
            f.write(s + '\n')
            for j in self.param[i]:
                s = '  {v1}:{v2}'.format(v1=j, v2=self.param[i][j])
                print(s)
                f.write(s + '\n')
        f.close()

    # 数据集加载
    def load_data(self):
        print('loding data')
        train_data, test_data, num_users, num_pois = load_data(self.param)
        self.param['dataset']['num_users'] = num_users
        self.param['dataset']['num_pois'] = num_pois
        self.trainset = POI_Dataset(train_data, self.param, train_flag=True)
        self.testset = POI_Dataset(test_data, self.param, train_flag=False)

        print('load data success')

    # 模型加载
    def load_model(self):
        print('loding model')
        if self.param['other']['test_only']:
            pass
        else:
            self.generator = Generator(self.param)
            self.discriminator = Discriminator(self.param)
        print('load model success')

    # 模型训练
    def train(self,loss_flag=False):
        # 初始化优化器、学习率和损失
        lr = self.param['other']['lr']
        G_optimizer = optim.SGD(self.generator.parameters(), lr=lr, momentum=0.9)
        D_optimizer = optim.SGD(self.discriminator.parameters(), lr=lr, momentum=0.9)
        BCE = nn.BCELoss(reduction='mean')
        MSE = nn.MSELoss(reduce=True, size_average=True)
        CE=nn.CrossEntropyLoss()
        criterion1 = nn.NLLLoss()
        criterion2 = nn.MSELoss()
        # 初始化模型device
        device = self.param['other']['device']
        self.generator.to(device)
        self.discriminator.to(device)
        # 初始化随机seed
        torch.manual_seed(self.param['other']['seed'])
        np.random.seed(self.param['other']['seed'])
        random.seed(self.param['other']['seed'])
        # 其他训练参数
        batch_size = self.param['other']['batch_size']
        alpha = self.param['other']['alpha']
        beta = self.param['other']['beta']
        # 数据加载
        train_loader = DataLoader(self.trainset, shuffle=True, batch_size=batch_size, drop_last=True)

        # 结果保存文件
        result_file = open(os.path.join(self.root, 'result.txt'), 'a+')
        # training in each epoch
        if loss_flag:
            D_loss_file=open('./'+self.param['dataset']['name']+'_alpha_'+str(self.param['other']['alpha'])+'_beta_'+str(self.param['other']['beta'])+'D_loss.csv','w',newline='')
            G_loss_file = open('./' + self.param['dataset']['name'] +'_alpha_'+str(self.param['other']['alpha'])+'_beta_'+str(self.param['other']['beta'])+ 'G_loss.csv','w',newline='')
            D_iteration=0
            G_iteration=0
            D_loss_writer=csv.writer(D_loss_file)
            G_loss_writer=csv.writer(G_loss_file)
        # pre_file=open(os.path.join(self.root,'pre.csv'),'w',newline='')
        # rec_file = open(os.path.join(self.root, 'rec.csv'), 'w', newline='')
        # f1_file = open(os.path.join(self.root, 'f1.csv'), 'w', newline='')
        # pre_writer=csv.writer(pre_file)
        # rec_writer=csv.writer(rec_file)
        # f1_writer=csv.writer(f1_file)
        for epoch in range(self.param['other']['num_epochs']):

            self.generator.train()
            self.discriminator.train()

            #             D-step
            D_train_loss = 0.0
            for i, data in enumerate(train_loader):
                user_tensor, context_tensor, label_tensor, context_cordi_tensor, label_cordi_tensor = data
                user_tensor = Variable(user_tensor.to(device))
                context_tensor = Variable(context_tensor.to(device))
                label_tensor = Variable(label_tensor.to(device))
                context_cordi_tensor = Variable(context_cordi_tensor.to(device))
                label_cordi_tensor = Variable(label_cordi_tensor.to(device))
                fake_pro, fake_cordi = self.generator(user_tensor, context_tensor, context_cordi_tensor)
                fake_pro = fake_pro.detach()
                D_output = self.discriminator(user_tensor, context_tensor, context_cordi_tensor, fake_pro)
                target = torch.zeros(batch_size, 1).to(device)
                D_loss1 = BCE(D_output, target)  # fake部分
                D_output = self.discriminator(user_tensor, context_tensor, context_cordi_tensor, label_tensor)
                target = torch.ones(batch_size, 1).to(device)
                D_loss2 = BCE(D_output, target)  # real部分
                D_loss = D_loss1 + D_loss2
                if loss_flag:
                    D_iteration+=1
                    if D_iteration%50==0:
                        line=[D_iteration,D_loss1.item(),D_loss2.item()]
                        D_loss_writer.writerow(line)
                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()
                D_train_loss += D_loss.item() / batch_size
            D_train_loss /= (i + 1)

            # G-step
            G_amb = 3
            G_train_loss = 0.0
            for i, data in enumerate(train_loader):
                user_tensor, context_tensor, label_tensor, context_cordi_tensor, label_cordi_tensor = data
                user_tensor = Variable(user_tensor.to(device))
                context_tensor = Variable(context_tensor.to(device))
                label_tensor = Variable(label_tensor.to(device))
                context_cordi_tensor = Variable(context_cordi_tensor.to(device))
                label_cordi_tensor = Variable(label_cordi_tensor.to(device))
                fake_pro, fake_cordi = self.generator(user_tensor, context_tensor, context_cordi_tensor)
                D_output = self.discriminator(user_tensor, context_tensor, context_cordi_tensor, fake_pro)
                # 判别误差
                target = torch.ones(batch_size, 1).to(device)
                G_loss_rf = BCE(D_output, target)

                # 预测误差
                # y_target=label_tensor.argmax(dim=1)
                # G_loss_rec= CE(fake_pro,y_target)
                # G_loss_rec=G_loss_rec/batch_size
                G_loss_rec = torch.nn.functional.log_softmax(fake_pro, dim=1)
                G_loss_rec = torch.mul(G_loss_rec, label_tensor)
                G_loss_rec = -1 * torch.sum(G_loss_rec) / batch_size

                # 回归误差
                G_loss_reg = MSE(fake_cordi, label_cordi_tensor)

                G_loss = alpha * G_loss_rec + beta * G_loss_reg + (1 - alpha - beta) * G_loss_rf

                if loss_flag:
                    G_iteration+=1
                    if G_iteration%50==0:
                        line=[G_iteration,G_loss_rec.item(),G_loss_reg.item(),G_loss_rf.item()]
                        G_loss_writer.writerow(line)

                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                G_train_loss += G_loss.item() / batch_size
            G_train_loss /= (i + 1)

            pre, rec, f1 = self.test()
            s1 = 'epoch:{v1},G_loss:{v2:.4f},D_loss:{v3:.4f}'.format(v1=epoch, v2=G_train_loss, v3=D_train_loss)
            print(s1)
            s2 = 'Precision:{v1} \nRecall:{v2} \nF1:{v3} \n'.format(v1=pre, v2=rec, v3=f1)
            print(s2)

            result_file.write(s1 + '\n')
            result_file.write(s2)
        result_file.close()

        # 保存模型参数
        if self.generator:
            generater_dict = self.generator.state_dict()
        if self.discriminator:
            discriminator_dict = self.discriminator.state_dict()
        model_file = os.path.join(self.root, 'checkpoint.pth.tar')
        state = {
            'model_name': self.param['model']['name'],
            'state_dict': [generater_dict, discriminator_dict]
        }
        self.save_checkpoint(state, model_file)

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    # 模型测试
    def test(self):
        device = self.param['other']['device']
        batch_size = self.param['other']['batch_size']
        TopN = self.param['other']['topN']
        cnt = 0  # 记录推荐次数
        self.generator.eval()
        test_loader = DataLoader(self.testset, batch_size=128, shuffle=False, drop_last=False)
        Precision = []
        Recall = []
        F1 = []
        Hits = [0, 0, 0, 0]
        all_visits = 0
        #         reclist_file=open(os.path.join(self.root,'reclist.csv'),'w',newline='')
        #         writer=csv.writer(reclist_file)
        for i, data in enumerate(test_loader):
            user_tensor, context_tensor, label_tensor, context_cordi_tensor, test_mask = data
            user_tensor = user_tensor.to(device)
            context_tensor = context_tensor.to(device)
            label_tensor = label_tensor
            context_cordi_tensor = context_cordi_tensor.to(device)
            pro, _ = self.generator(user_tensor, context_tensor, context_cordi_tensor)
            pro = pro.cpu()
            cnt += label_tensor.size()[0]
            all_visits += torch.sum(label_tensor).item()

            for j in range(len(TopN)):
                k = TopN[j]
                #                 pro=torch.mul(pro,test_mask)
                _, rec = torch.topk(pro, k)
                predict = torch.zeros(rec.size()[0], self.param['dataset']['num_pois']).scatter_(1, rec, 1)
                hit = torch.sum(torch.mul(predict, label_tensor))
                hit = int(hit.item())
                Hits[j] += hit

        #             rec=torch.topK(pro,50)
        #             rec=rec.data.numpy().tolist()
        #             target=label_tensor.data.numpy().tolist()
        #             for j in range(len(rec)):
        #                 cnt+=1
        #                 recommend=rec[j]
        #                 real_visit=target[j]
        #                 all_visits+=len(real_visit)
        #                 writer.writerow(recommend)
        #                 hit=0

        #                 for k in len(recommend):
        #                     if k in TopN:
        #                         index=TopN.index(k)
        #                         Hits[index]+=hit
        #                     if recommend[k] in real_visit:
        #                         hit+=1
        #         reclist_file.close()
        for i in range(len(TopN)):
            K = TopN[i]
            pre = float(Hits[i]) / (K * self.param['dataset']['num_users'])
            rec = float(Hits[i]) / (all_visits)
            f1 = 2 * pre * rec / (rec + pre)
            Precision.append(format(pre, '.4f'))
            Recall.append(format(rec, '.4f'))
            F1.append(format(f1, '.4f'))
        print(all_visits)
        return Precision, Recall, F1


para_meta = {
    'dataset':{
        'name':'Foursquare',
        'num_users':1895,
        'num_pois':2109,
        'embedding_url':None,
    },
    'model':{
        'name':'POI_GAN',
        'embedding_dim':200,
        'hidden_size':200,
        'LSTM_layer':2
    },
    'other':{
        'experiment_name':'Foursquare_std',
        'device':torch.device("cuda:0"),
#         'device':torch.device("cpu"),
        'test_only':False,
        'split_rate':0.85,
        'sliding_window':3,
        'topN':[5, 10, 15, 20],
        'seed':1,
        'batch_size':32,
        'num_epochs':20,
        'lr':0.01,
        'alpha':0.5,
        'beta':0.3
    }
}

def main():

    para_meta['dataset']['name'] = 'Foursquare'
    para_meta['other']['experiment_name'] = para_meta['dataset']['name']
    model = POI_Recommendation_Experiment(para_meta)
    model.train()


if __name__ == '__main__':
    main()





