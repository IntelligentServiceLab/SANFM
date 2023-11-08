import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn as nn
from sanfm import SANFM
from sanfm import train
from sanfm import tst

def main():
    # load data
    data = pd.read_csv('./input_bert_data.csv')

    sparse_features = ['1538', '1539']  #该特征为数据集中“离散特征”所在列数。数据集中每行为一个完整数据，包含首列：标签，2-1537列：连续特征，1538-1539列：离散特征。
    dense_features = [str(i) for i in range(2, 1538)]   #连续特征由bert模型处理mashup和api的文本描述获得的768维数值向量拼接构成，共两个，计1536维

    data[sparse_features] = data[sparse_features].fillna('-1', )    #填充-1和0，下同
    data[dense_features] = data[dense_features].fillna(0, )
    # target = ['1']  # 数据读取完毕
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features 对稀疏特征进行标签编码，对密集特征进行简单变换
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # mms_1 = MinMaxScaler(feature_range=(0, 1))
    # mms_2 = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms_1.fit_transform(data[dense_features])  # 标签进行数值编码，数值进行minmax化
    # data[sparse_features] = mms_2.fit_transform(data[sparse_features])    #我觉得根本不需要这玩意
    # 2.格式化数据
    data_np = np.array(data)
    data_np1, data_np2 = np.split(data_np, [1], axis=1)  # 拆分成两个array，tensor化后可以直接接tensordata
    x_train, x_test, y_train, y_test = train_test_split(data_np2, data_np1, test_size=0.4, random_state=1)

    x_train_tensor = torch.tensor(x_train)
    x_test_tensor = torch.tensor(x_test)
    y_train_tensor = torch.tensor(y_train)
    y_test_tensor = torch.tensor(y_test)

    train_tensor_set = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_tensor_set, batch_size=256, shuffle=True)
    test_tensor_set = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_tensor_set, batch_size=256, shuffle=True)

    # 实例化模型
    # loss_train = np.inf
    sanfm_obj = SANFM(embed_dim=32, droprate = 0.5, i_num = 1536, c_num = 2)    #注意此处i_num和c_num分别对应连续特征维数和离散特征维数
    optimizer = torch.optim.Adam(sanfm_obj.parameters(), lr=0.001)
    LOSS_total = np.inf
    AUC_total = np.inf
    endure_count = 0

    for epoch in range(250):
        # train
        train(sanfm_obj, train_loader, optimizer, epoch)
        # test
        LOSS, AUC = tst(sanfm_obj, test_loader)

        if LOSS_total > LOSS:  # loss_train:
            LOSS_total = LOSS  # loss_train
            AUC_total = AUC
            endure_count = 0
        else:
            endure_count += 1

        print("<Test> LOSS: %.5f AUC: %.5f" % (LOSS, AUC))

        if endure_count > 30:
            break
    LOSS, AUC = tst(sanfm_obj, test_loader)


    print('The best LOSS: %.5f AUC: %.5f' % (LOSS, AUC))

if __name__ == "__main__":
    main()