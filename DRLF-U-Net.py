import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import torch
import numpy as np
from numpy import *
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

class SEBlock(nn.Module):

    def __init__(self,ch_in):
        super(SEBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # N * 32 * 1 * 1
        self.fc1 = nn.Linear(in_features = int(ch_in), out_features = int(ch_in//2))
        self.fc2 = nn.Linear(in_features = int(ch_in//2), out_features = int(ch_in))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # sequeeze
        out = self.global_pool(x)   
        out = out.view(out.size(0), -1)
        # Excitation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        # Scale
        # out = out * x
        # out += x
        # out = self.relu(out)

        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

# 计算距离（欧几里得）
def euclDistance(vector1, vector2):
    # vector1 = vector1.cuda()
    a = vector2 - vector1
    return torch.sqrt(torch.sum(a ** 2))

# 初始化质心
def initCentroids(data, k):
    numSamples, dim = data.shape
    # k个质心，列数跟样本的列数一样
    centroids = torch.zeros((k, dim))
    # 随机选出k个质心
    for i in range(k):
        # 随机选取一个样本的索引
        index = int(torch.Tensor(1,1).uniform_(0, numSamples))
        # index = int(torch.Tensor(index))
        # 作为初始化的质心
        # centroids[i, :] = data[index, :].detach()
        centroids[i, :] = data[index, :]
    return centroids

# k-means算法实现过程
# 传入数据集和k值
def kmeans(data, k):
    # 计算样本个数
    # data = [2, 512]
    numSamples = data.shape[0]  # 2
    # numSamples = torch.sh.shape(data)[0]  # 2
    # 样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    clusterData = torch.zeros((numSamples, 2)) # 
    # 决定质心是否要改变的质量
    clusterChanged = True
    # 初始化质心
    centroids = initCentroids(data, k)
    # centroids = torch.tensor(centroids)
    while clusterChanged:
        clusterChanged = False
        # 循环每一个样本
        for i in range(numSamples):
            # 最小距离
            minDist = 100000.0
            # 定义样本所属的簇
            minIndex = 0
            # 循环计算每一个质心与该样本的距离
            for j in range(k):
                # 循环每一个质心和样本，计算距离
                distance = euclDistance(centroids[j, :], data[i, :])
                # 如果计算的距离小于最小距离，则更新最小距离
                if distance < minDist:
                    minDist = distance
                    # 更新最小距离
                    clusterData[i, 1] = minDist
                    # 更新样本所属的簇
                    minIndex = j
            # 如果样本的所属的簇发生了变化
            if clusterData[i, 0] != minIndex:
                # 质心要重新计算
                clusterChanged = True
                # 更新样本的簇
                clusterData[i, 0] = minIndex
        # 更新质心
        for j in range(k):
            # 获取第j个簇所有的样本所在的索引
            cluster_index = torch.nonzero(clusterData[:, 0] == j)
            # 第j个簇所有的样本点
            pointsInCluster = data[cluster_index]
            # 计算质心
            centroids[j, :] = torch.mean(pointsInCluster, axis=0)
        # cluster = [j,cluster_index]
    return clusterData
class Cnet(nn.Module):
    def __init__(self):
        super(Cnet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(1, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(8, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(16, 32)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(32, 64)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(64, 128)

        self.conv1_1 = DoubleConv(4, 1)
        self.pool1_1 = nn.MaxPool2d(2)
        self.conv2_1 = DoubleConv(1, 1)
        self.pool2_1 = nn.MaxPool2d(2)
        self.conv3_1 = DoubleConv(1, 1)
        self.pool3_1 = nn.MaxPool2d(2)
        self.conv4_1 = DoubleConv(1, 1)
        self.pool4_1 = nn.MaxPool2d(2)
        self.conv5_1 = DoubleConv(1, 1)
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        self.up7 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv7 = DoubleConv(64, 32)
        self.up8 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv8 = DoubleConv(32, 16)
        self.up9 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.conv9 = DoubleConv(16, 8)
        self.conv10 = nn.Conv2d(8, 3, 1)
        
        self.R1 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16
        self.R1_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16
        self.RP2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16
        self.RP2_1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16

        self.RP3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )
        self.RP3_1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )
        self.RP4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )

        self.RP4_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )
        self.RP5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )

        self.RP5_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )

        self.R2_v1 = nn.Sequential(
            nn.Conv2d(1, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.R3_v1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.R4_v1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.R5_v1 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.R6_v1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.R7_v1 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.R8_v1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.R2_v2 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.R3_v2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.R4_v2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.R5_v2 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.R6_v2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.R7_v2 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.R8_v2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.SE1 = SEBlock(8)
        self.SE2 = SEBlock(16)
        self.SE3 = SEBlock(32)
        self.SE4 = SEBlock(64)
        self.SE5 = SEBlock(128)
        self.SP1 = SpatialAttention(64)
        self.SP2 = SpatialAttention(128)
        self.SP3 = SpatialAttention(256)
        self.SP4 = SpatialAttention(512)
        self.SP5 = SpatialAttention(1024)
        self.fc1 = nn.Linear(in_features=12288, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=12800, out_features=8)

    def forward(self, MS, PAN):
       
        
        up = torch.nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None)
        MS = up(MS)

        c1_MS  = self.R1(MS)        # [2, 64, 64, 64]
        c1_SE_MS  = self.SE1(c1_MS) # [2, 64,  1,  1]
        c1_PAN = self.conv1(PAN)     # [2, 64, 64, 64]
        c1_PAN_EN = c1_SE_MS * c1_PAN  # [2, 64, 64, 64]
        p1_PAN = self.pool1(c1_PAN_EN)  # [2, 64, 32, 32]
    
        c2_MS = self.RP2(c1_MS)    # [2, 128, 32, 32]
        c2_SE_MS = self.SE2(c2_MS) # [2, 128,  1,  1]
        c2_PAN = self.conv2(p1_PAN)  # [2, 128, 32, 32]
        c2_PAN_EN = c2_SE_MS * c2_PAN  # [2, 128, 32, 32]
        p2_PAN = self.pool2(c2_PAN_EN)  # [2, 128, 16, 16]

        c3_MS = self.RP3(c2_MS)    # [2, 256, 16, 16]
        c3_SE_MS = self.SE3(c3_MS) # [2, 256,  1,  1]
        c3_PAN = self.conv3(p2_PAN)  # [2, 256, 16, 16]
        c3_PAN_EN = c3_SE_MS * c3_PAN  # [2, 256, 16, 16]
        p3_PAN = self.pool3(c3_PAN_EN)  # [2, 256, 8, 8]

        c4_MS = self.RP4(c3_MS)    # [2, 512, 8, 8]
        c4_SE_MS= self.SE4(c4_MS) # [2, 512,  1,  1]
        c4_PAN = self.conv4(p3_PAN)  # [2, 512, 8, 8]
        c4_PAN_EN = c4_SE_MS * c4_PAN  # [2, 512, 8, 8]
        p4_PAN = self.pool4(c4_PAN_EN)  # [2, 512, 4, 4]

        c5_MS = self.RP5(c4_MS)    # [2, 1024, 4, 4]
        c5_SE_MS = self.SE5(c5_MS) # [2, 1024,  1,  1]
        c5_PAN = self.conv5(p4_PAN)  # [2, 1024, 4, 4]
        c5_PAN_EN = c5_SE_MS * c5_PAN  # [2, 1024, 4, 4]

        # 用PAN的空间信息去增强MS的空间信息
        c1_PAN_1 = self.R1_1(PAN)        # [2, 64, 64, 64]
        c1_SP_PAN = self.SP1(c1_PAN_1)   # [2, 1,  64,  64]
        c1_MS_1 = self.conv1_1(MS)       # [2, 1, 64, 64]
        c1_MS_EN = c1_SP_PAN * c1_MS_1      # [2, 1, 64, 64]
        p1_MS = self.pool1_1(c1_MS_EN)    # [2, 1, 32, 32]

        c2_PAN_1 = self.RP2_1(c1_PAN_1)    # [2, 128, 32, 32]
        c2_SP_PAN = self.SP2(c2_PAN_1)   # [2, 1,  32,  32]
        c2_MS_1 = self.conv2_1(p1_MS)    # [2, 1, 32, 32]
        c2_MS_EN = c2_SP_PAN * c2_MS_1      # [2, 1, 32, 32]
        p2_MS = self.pool2_1(c2_MS_EN)    # [2, 1, 16, 16]

        c3_PAN_1 = self.RP3_1(c2_PAN_1)    # [2, 256, 16, 16]
        c3_SP_PAN = self.SP3(c3_PAN)   # [2, 1,  16,  16]
        c3_MS_1 = self.conv3_1(p2_MS)    # [2, 1, 16, 16]
        c3_MS_EN = c3_SP_PAN * c3_MS_1      # [2, 1, 16, 16]
        p3_MS = self.pool3_1(c3_MS_EN)    # [2, 1, 8, 8]

        c4_PAN_1 = self.RP4_1(c3_PAN_1)    # [2, 512, 8, 8]
        c4_SP_PAN= self.SP4(c4_PAN_1)    # [2, 1,  8,  8]
        c4_MS_1 = self.conv4_1(p3_MS)    # [2, 1, 8, 8]
        c4_MS_EN = c4_SP_PAN * c4_MS_1      # [2, 1, 8, 8]
        p4_MS = self.pool4_1(c4_MS_EN)    # [2, 1, 4, 4]

        c5_PAN_1 = self.RP5_1(c4_PAN_1)    # [2, 1024, 4, 4]
        c5_SP_PAN = self.SP5(c5_PAN_1)   # [2, 1, 4, 4]
        c5_MS_1 = self.conv5_1(p4_MS)    # [2, 1, 4, 4]
        c5_MS_EN = c5_SP_PAN * c5_MS_1      # [2, 1, 4, 4]

        # 信息分离
        # MS信息分离
        c5_MS_5 = self.R2_v1(c5_MS_EN)
        c5_MS_3 = self.R3_v1(c5_MS_EN)
        c5_MS_5 = self.R4_v1(c5_MS_5)
        c5_MS_3 = self.R5_v1(c5_MS_3)
        c5_MS_5 = self.R6_v1(c5_MS_5)
        c5_MS_3 = self.R7_v1(c5_MS_3)

        # PAN信息分离
        c5_PAN_5 = self.R2_v2(c5_PAN_EN)
        c5_PAN_3 = self.R3_v2(c5_PAN_EN)
        c5_PAN_5 = self.R4_v2(c5_PAN_5)
        c5_PAN_3 = self.R5_v2(c5_PAN_3)
        c5_PAN_5 = self.R6_v2(c5_PAN_5)
        c5_PAN_3 = self.R7_v2(c5_PAN_3)
        c5_PAN_SP = torch.cat([c5_PAN_3,c5_PAN_5],1)
        c5_PAN_SP = self.R8_v2(c5_PAN_SP)

        semantic_loss = c5_MS_5 - c5_PAN_5
        semantic_loss = torch.tanh(torch.norm(semantic_loss))
        detail_loss = c5_MS_3 - c5_PAN_3
        detail_loss = torch.tanh(torch.norm(detail_loss))

        up_6 = self.up6(c5_PAN_SP)  # [2, 512, 64, 64]
        merge6 = torch.cat([up_6, c4_PAN], dim=1) # [2, 1024, 64, 64]
        c6 = self.conv6(merge6) # [2, 512, 64, 64]
        up_7 = self.up7(c6)     # [2, 256, 128, 128]

        merge7 = torch.cat([up_7, c3_PAN], dim=1) # [2, 512, 128, 128]
        c7 = self.conv7(merge7) # [2, 256, 128, 128]
        up_8 = self.up8(c7)     # [2, 128, 256, 256]

        merge8 = torch.cat([up_8, c2_PAN], dim=1) # [2, 256, 256, 256]
        c8 = self.conv8(merge8) # [2, 128, 256, 256]
        up_9 = self.up9(c8)     # [2, 64, 512, 512]

        merge9 = torch.cat([up_9, c1_PAN], dim=1) # [2, 128, 512, 512]
        c9 = self.conv9(merge9) # [2, 64, 512, 512]
        out = self.conv10(c9)   # [2, 3, 512, 512]
        Local_Feature = out
        # out = nn.Sigmoid()(c10) # [2, 3, 512, 512]

        # 全连接分类
        input_FC = out.view(out.shape[0], -1)
        input_FC = F.relu(self.fc1(input_FC))
        input_FC = F.relu(self.fc2(input_FC))

        Global_Feature = input_FC

        k = 8
        clusterData = kmeans(Global_Feature, k)
        #  得到聚类的索引
        clustert_list = []
        for j in range(k):
            # 获取第j个簇所有的样本所在的索引
            clustert_list.append(torch.nonzero(clusterData[:, 0] == j))

        n,c,h,w = Local_Feature.size()
        Sim_array = torch.zeros(n, h, w)
        for i in clustert_list:
            for j in i:
                
                length = len(j)
                if(length != 0):
                    for k in j:
                        Local_C = torch.zeros(c, h, w)
                        # Local_C = Local_C.cuda()
                        Local_C += Local_Feature[k,::]
                    Local_C = Local_C / length                                    # 512*512*3

                    for k in j:
                        Sim_array[k, ::] = torch.cosine_similarity(Local_C,Local_Feature[k,::],0)  # 512*512

        Sim = torch.reshape(Sim_array, (n, h * w))
        _, indices = torch.sort(Sim, dim = 1, descending=True)   

        Local_Feature_reshape = torch.reshape(Local_Feature,(n, c, h*w))
        temp = torch.unsqueeze(indices, dim=1)
        temp = temp.expand(n, c, h*w)
        # temp = temp.cuda()
        Local_Feature_reshape_sort = Local_Feature_reshape.gather(dim = 2, index = temp)
        Local_Feature_reshape_sort = Local_Feature_reshape_sort.view(Local_Feature_reshape_sort.shape[0], -1)

        input_FC = torch.cat((Local_Feature_reshape_sort, Global_Feature), 1)
        
        out_FC = self.fc3(input_FC)

        return out_FC, semantic_loss, detail_loss

if __name__ == "__main__":
    PAN = torch.randn(16, 1, 64, 64)
    MS = torch.randn( 16, 4, 16, 16)
    CNet = Cnet()
    out_result,semantic_loss, detail_loss = CNet(MS,PAN)
    print(out_result)
    print(out_result.shape)
    print(semantic_loss,semantic_loss.shape)
    print(detail_loss,detail_loss.shape)
