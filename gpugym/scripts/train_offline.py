from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch
import numpy as np
import torch
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

import torch.share

logger = Logger(0.001)
dict_list = []
dof_pos_key = []
dof_tau_key = []
dof_vel_key = []
for j in range(12):
    dof_pos_key.append('dof_pos'+'_'+str(j+1))
    dof_vel_key.append('dof_vel'+'_'+str(j+1))
    dof_tau_key.append('dof_torque'+'_'+str(j+1))

gravatiy_key = ['gravity_x','gravity_y','gravity_z']
base_v_key = ['vx','vy','vz']
base_w_key = ['wx','wy','wz']
contact_F_key = ['contact_forces_z_L','contact_forces_z_R']


filepath = '/home/xie/Desktop/pbrs-humanoid/experiment/play_log.csv'
dict_list= logger.read_dicts_from_csv(filepath)
for key,value in dict_list[0].items():
    print(key,value)
print("dict_list", len(dict_list))
data_all = len(dict_list)-1
obs_num = len(dof_pos_key)+len(dof_tau_key)+len(dof_tau_key)+len(gravatiy_key)\
+len(base_w_key)+len(contact_F_key)
est_num = 3+1#vx,vy,vz,hz
obs_key = contact_F_key+base_w_key+gravatiy_key+dof_tau_key+dof_tau_key+dof_pos_key
est_key = base_v_key+['height_z']
feature = np.zeros((data_all,obs_num))
label = np.zeros((data_all,est_num))
for i in range(data_all):
    for j in range(len(obs_key)):
        feature[i,j] = dict_list[i][obs_key[j]]
    for k in range(len(est_key)):
        label[i,k] = dict_list[i][est_key[k]]
print("feature",feature[0,:])
print("label",label[0,:])

print("feature",np.shape(feature))
print("label",np.shape(label))

features = torch.tensor(feature, dtype=torch.float32)
labels = torch.tensor(label, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = 'cpu'
# 将数据打包到一个元组中
data_arrays = (features, labels)
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(44, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)
torch.manual_seed(0)
np.random.seed(0)

model = MLP()
model.to(device)
features.to(device)
labels.to(device)
data_iter = load_array((features, labels), batch_size)
# data_iter.to(device)
train = 1

if train:
# 定义损失函数和优化器
    criterion = nn.MSELoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            # 前向传播
            outputs = model(x)
            # outputs.to(device)
            # 计算损失
            loss = criterion(outputs, y)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        l = criterion(model(x), y)
        print(f'epoch {epoch + 1}, loss {l:f}')
    # 绘制原始数据和拟合曲线

    torch.save(model.state_dict(), 'mlp.params')
    md = model
    md.to(device)

else:
    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()
    md = clone
    md.to(device)

t = np.arange(0, 9.99, 0.01)
md.to('cpu')
print(features.device)
# print()
# features.to('cuda')
y_est = md(features).detach().numpy()

print("features",features.size())
print("features",md(features).size())

plot = 0
if plot:
    # 绘制第一个曲线
    plt.figure(1)
    plt.plot(t, label[:,0], label='vx')
    plt.plot(t, y_est[:,0], label='vx_est')
    plt.legend()

    plt.grid(True)

    # 绘制第二个曲线
    plt.figure(2)
    plt.plot(t, label[:,1], label='vy')
    plt.plot(t, y_est[:,1], label='vy_est')
    plt.legend()

    plt.grid(True)

    # 绘制第三个曲线
    plt.figure(3)
    plt.plot(t, label[:,2], label='vz')
    plt.plot(t, y_est[:,2], label='vz_est')
    plt.legend()

    plt.grid(True)

    # 绘制第四个曲线
    plt.figure(4)
    plt.plot(t, label[:,3], label='HZ')
    plt.plot(t, y_est[:,3], label='hz_est')

    plt.legend()
    plt.grid(True)

    # 显示所有图形
    plt.show()


print(torch.cuda.device_count())




