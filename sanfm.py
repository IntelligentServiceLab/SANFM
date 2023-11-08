import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn as nn

class SANFM(nn.Module):
    def __init__(self, embed_dim, droprate = 0.5, i_num = 1536, c_num = 2):
        super(SANFM, self).__init__()

        self.i_num = i_num
        self.c_num = c_num
        self.embed_dim = embed_dim  #暂定64
        self.att_dim = embed_dim  #暂定，用于selfatt的输出维度，可修改
        self.bi_inter_dim = embed_dim  #暂定，可修改
        self.droprate = droprate
        self.criterion = nn.BCELoss(weight=None, reduction='mean')
        self.sigmoid = nn.Sigmoid()

        self.dense_embed = nn.Linear((self.i_num + self.c_num), self.embed_dim) #将batch*1500+的转化为batch*64，一行为64维向量，达到dense的目的
        
        self.pairwise_inter_v = nn.Parameter(torch.empty(self.embed_dim, self.bi_inter_dim))  #用于pairwise interaction
        
        #以下为selfatt所需参数
        self.query_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.key_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.value_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.softmax = nn.Softmax(dim=-1)	#当nn.Softmax的输入是一个二维张量时，其参数dim = 0，是让列之和为1；dim = 1，是让行之和为1
        #以上为selfatt所需参数

        self.hidden_1 = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.hidden_2 = nn.Linear(self.embed_dim, 1)

        self.bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self._init_weight_()

    def BiInteractionPooling(self, pairwise_inter):
        inter_part1_sum = torch.sum(pairwise_inter, dim=1)
        inter_part1_sum_square = torch.square(inter_part1_sum)  # square_of_sum

        inter_part2 = pairwise_inter * pairwise_inter
        inter_part2_sum = torch.sum(inter_part2, dim=1)  # sum of square
        bi_inter_out = 0.5 * (inter_part1_sum_square - inter_part2_sum)
        return bi_inter_out


    def _init_weight_(self):
        """ Try to mimic the original weight initialization. parameter不需要在此处初始化"""
        # dense embedding
        nn.init.normal_(self.dense_embed.weight, std=0.01)
        # pairwise interaction pooling
        nn.init.normal_(self.pairwise_inter_v, std=0.01)
        # deep layers
        nn.init.kaiming_normal_(self.hidden_1.weight)
        nn.init.kaiming_normal_(self.hidden_2.weight)
        # attention part
        nn.init.kaiming_normal_(self.query_matrix)
        nn.init.kaiming_normal_(self.key_matrix)
        nn.init.kaiming_normal_(self.value_matrix)

    def forward(self, batch_data):  #输出batch*103
        batch_data = batch_data.to(torch.float32)

        #embedding part
        dense_embed = self.dense_embed(batch_data)
        #interaction part
        pairwise_inter = dense_embed.unsqueeze(1) * self.pairwise_inter_v   #3d
        #print('pairwise_inter shape:', pairwise_inter.shape)
        pooling_out = self.BiInteractionPooling(pairwise_inter) #2d
        #print('pooling_out shape:', pooling_out.shape)
        
        #以下为自注意力
        X = pooling_out
        proj_query = torch.mm(X, self.query_matrix)	#把原先tensor中的数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor
        proj_key = torch.mm(X, self.key_matrix)
        proj_value = torch.mm(X, self.value_matrix)

        S = torch.mm(proj_query, proj_key.T)
        attention_map = self.softmax(S) #这里只是q*k

        # Self-Attention Map
        value_weight = proj_value[:,None] * attention_map.T[:,:,None]
        value_weight_sum = value_weight.sum(dim=0)  #此处认为是已经加权过了的原数据值，可以直接用于下一步。
        #以上为自注意力
        
        #MLP part
        mlp_hidden_1 = F.relu(self.bn(self.hidden_1(value_weight_sum)))   #暂时不用Batch Normalization，更新，感觉必须得用，不然loss维持在0.69，娘希匹
        mlp_hidden_2 = F.dropout(mlp_hidden_1, training=self.training, p=self.droprate)
        mlp_out = self.hidden_2(mlp_hidden_2)
        final_sig_out = self.sigmoid(mlp_out)
        final_sig_out_squeeze = final_sig_out.squeeze()
        return final_sig_out_squeeze

    def loss(self, batch_input, batch_label):
        pred = self.forward(batch_input)
        pred = pred.to(torch.float32)
        batch_label = batch_label.to(torch.float32).squeeze()
        loss1 = self.criterion(pred, batch_label)   #暂时不用regularization
        return loss1


def train(model, train_loader, optimizer, epoch):
    model.train()
    # global loss_train
    avg_loss = 0.0

    for i, data in enumerate(train_loader):
        batch_input, batch_label = data
        optimizer.zero_grad()
        loss2 = model.loss(batch_input, batch_label)
        loss2.backward(retain_graph = True)
        optimizer.step()

        avg_loss += loss2.item()

        if (i + 1) % 10 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 10))
            # loss_train = avg_loss
            avg_loss = 0.0
    return 0

# def tst(model, test_loader):
#     model.eval()
#     LOSS = []
#     AUC = []
#
#     for test_input, test_label in test_loader:
#         pred = model(test_input)
#         pred = pred.to(torch.float32)
#         test_label = test_label.squeeze().to(torch.float32)
#         loss_value = log_loss(test_label.detach().tolist(), pred.detach().tolist())
#         auc_value = roc_auc_score(test_label.detach().tolist(), pred.detach().tolist())
#         LOSS.append(loss_value)
#         AUC.append(auc_value)
#     loss = np.mean(LOSS)
#     auc = np.mean(AUC)
#     return loss, auc

def tst(model, test_loader):
    model.eval()
    criterion = nn.BCELoss(reduction='mean')
    LOSS = []
    AUC = []

    for test_input, test_label in test_loader:
        pred = model(test_input)
        pred = pred.to(torch.float32)
        test_label = test_label.squeeze().to(torch.float32)
        loss_value = criterion(pred, test_label)
        auc_value = roc_auc_score(test_label.detach().tolist(), pred.detach().tolist())
        LOSS.append(loss_value.detach())
        AUC.append(auc_value)
    loss = np.mean(LOSS)
    auc = np.mean(AUC)
    return loss, auc


