from utils import World, Rectangle
import numpy as np
import matplotlib.pyplot as plt
import random

"""
OccGrids_1 = []
for i in range(19520):
    if i%1000 == 0:
        print(i)
    OccGrids_1.append(np.loadtxt("OccGrids_1/OccGrids_1_"+str(i)+".txt"))
OccGrids_1 = np.array(OccGrids_1)
OccGrids_1 = OccGrids_1.reshape(488*40, 10, 54)
# print(OccGrids_1.shape)

OccGrids_2 = []
for i in range(15280):
    if i%1000 == 0:
        print(i)
    OccGrids_2.append(np.loadtxt("OccGrids_2/OccGrids_2_"+str(i)+".txt"))
OccGrids_2 = np.array(OccGrids_2)
OccGrids_2 = OccGrids_2.reshape(382*40, 10, 54)
# print(OccGrids_2.shape)


OccGrids_3 = []
for i in range(16160):
    if i%1000 == 0:
        print(i)
    OccGrids_3.append(np.loadtxt("OccGrids_3/OccGrids_3_"+str(i)+".txt"))
OccGrids_3 = np.array(OccGrids_3)
OccGrids_3 = OccGrids_3.reshape(404*40, 10, 54)
# print(OccGrids_3.shape)


change_lane_label_1 = np.loadtxt("change_lane_label_1.txt")
new_host_1 = np.loadtxt("new_host_1.txt")
host_1 = np.loadtxt("host_1.txt")
init_goal_1 = np.loadtxt("init_goal_1.txt")
init_goal_1 = np.array(init_goal_1).reshape(488*31,2,4)
# print(init_goal_1.shape)
new_host_1 = np.array(new_host_1).reshape(488,40,4)
# print(new_host_1.shape)

change_lane_label_2 = np.loadtxt("change_lane_label_2.txt")
new_host_2 = np.loadtxt("new_host_2.txt")
host_2 = np.loadtxt("host_2.txt")
init_goal_2 = np.loadtxt("init_goal_2.txt")
init_goal_2 = np.array(init_goal_2).reshape(382*31,2,4)
# print(init_goal_2.shape)
new_host_2 = np.array(new_host_2).reshape(382,40,4)
# print(new_host_2.shape)


change_lane_label_3 = np.loadtxt("change_lane_label_3.txt")
new_host_3 = np.loadtxt("new_host_3.txt")
host_3 = np.loadtxt("host_3.txt")
# init_goal_3 = np.loadtxt("init_goal_3.txt")
# init_goal_3 = np.array(init_goal_3).reshape(404*31,2,4)
# print(init_goal_3.shape)
new_host_3 = np.array(new_host_3).reshape(404,40,4)
# print(new_host_3.shape)
init_goal_3 = []
for v in new_host_3:
    for f in range(31):
        init_goal_3.append([v[f], v[f+9]])
init_goal_3 = np.array(init_goal_3)


# train_OccGrids = np.concatenate((OccGrids_1, OccGrids_2))
test_OccGrids = OccGrids_3
# train_change_lane_label = np.concatenate((change_lane_label_1, change_lane_label_2))
test_change_lane_label = change_lane_label_3
# train_host = np.concatenate((host_1, host_2))
test_host = host_3
# train_new_host = np.concatenate((new_host_1, new_host_2))
test_new_host = new_host_3
# train_init_goal = np.concatenate((init_goal_1, init_goal_2))
test_init_goal = init_goal_3
# print("train:")
# print("Occ:",train_OccGrids.shape)
# print("label:",train_change_lane_label.shape)
# print("host:",train_host.shape)
# print("new host:",train_new_host.shape)
# print("init goal:",train_init_goal.shape)
print("test:")
print("Occ:",test_OccGrids.shape)
print("label:",test_change_lane_label.shape)
print("host:",test_host.shape)
print("new host:",test_new_host.shape)
print("init goal:",test_init_goal.shape)
"""

OccGrids_2 = []
for i in np.arange(204*31, 404*31):
    if i%1000 == 0:
        print(i)
    Occ = []
    for j in range(10):
        Occ.append(np.loadtxt("OccGrids_3_10f/OccGrids_3_" + str(i) + "-"+str(j)+".txt"))
    OccGrids_2.append(Occ)
OccGrids_2 = np.array(OccGrids_2)
print(OccGrids_2.shape)
OccGrids_2 = OccGrids_2.reshape(6200, 10, 10, 54)

change_lane_label_2 = np.loadtxt("change_lane_label_3.txt")
new_host_2 = np.loadtxt("new_host_3.txt")
host_2 = np.loadtxt("host_3.txt")
init_goal_2 = np.loadtxt("init_goal_3.txt")
init_goal_2 = np.array(init_goal_2).reshape(404*31,2,4)
# print(init_goal_2.shape)
new_host_2 = np.array(new_host_2).reshape(404,40,4)
# print(new_host_2.shape)
# init_goal_2 = []
# for v in new_host_2:
#     for f in range(31):
#         init_goal_2.append([v[f], v[f+9]])
# init_goal_2 = np.array(init_goal_2)

test_OccGrids = OccGrids_2
test_change_lane_label = change_lane_label_2[204:404]
test_host = host_2[204*41:404*41]
test_new_host = new_host_2[204:404]
test_init_goal = init_goal_2[204*31:404*31]
print("test:")
print("Occ:",test_OccGrids.shape)
print("label:",test_change_lane_label.shape)
print("host:",test_host.shape)
print("new host:",test_new_host.shape)
print("init goal:",test_init_goal.shape)



import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CondDataset(torch.utils.data.Dataset):
    def __init__(self, occ_grid, x, transform=None):
        self.data = occ_grid
        self.x = x
        self.transform = transform

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        if self.transform is not None:
            img = self.transform(self.data[index])
        else:
            img = self.data[index]
        return img, self.x[index]

    def __len__(self):
        return len(self.data)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=10,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            # nn.Conv2d(20, 16, 5, 27),
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.cnnout = nn.Linear(832, 128)  # fully connected layer, output 24 classes

    def forward(self, c):
        # c = c.type(torch.FloatTensor)
        c = self.conv1(c)
        c = self.conv2(c)
        c = c.view(c.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        c = self.cnnout(c)

        return c


class CNNEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128 + 4 + 4 + 44, 512)
        self.droplayer = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 128)

        self.linear_means = nn.Linear(128, 44)
        self.linear_log_var = nn.Linear(128, 44)

    def forward(self, x, c, init_goal):
        x = torch.cat((x, c, init_goal), dim=1)

        x = F.relu(self.fc1(x))
        x = self.droplayer(x)
        x = F.relu(self.fc2(x))

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class CNNDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128 + 4 + 4 + 44, 512)
        self.droplayer = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 128)

        self.out = nn.Linear(128, 44)

    def forward(self, z, c, init_goal):
        z = torch.cat((z, c, init_goal), dim=1)

        z = F.relu(self.fc1(z))
        z = self.droplayer(z)
        z = F.relu(self.fc2(z))

        x = self.out(z)

        return x


class CNNCVAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()
        self.cnn = CNNModel()

    def forward(self, x, c, init_goal):
        batch_size = x.size(0)

        c = self.cnn(c)

        means, log_var = self.encoder(x, c, init_goal)

        eps = torch.randn([batch_size, 44])
        z = (means + torch.exp(log_var / 2) * eps)

        recon_x = self.decoder(z, c, init_goal)

        return recon_x, means, log_var, z

    def inference(self, c, init_goal, n=1):
        batch_size = n
        z = torch.randn([batch_size, 44])

        c = self.cnn(c)

        recon_x = self.decoder(z, c, init_goal)

        return recon_x


test_Occ = []
test_i_g = []
test_x = []


i = 0
while i < len(test_OccGrids):
    if i%100 == 0:
        print(i)
    v_n = int(i/31)
    f_n = i%31
    # if change_lane_label[v_n] == 1:
    local_y = test_new_host[v_n][f_n][1]
    this_host = test_host[v_n*41 + f_n]
    if this_host[13] == 1:
        x_0 = 0-1
        x_1 = 40-1
    if this_host[13] == 2:
        x_0 = 0-1
        x_1 = 40-1
    if this_host[13] == 3:
        x_0 = 10-1
        x_1 = 50-1
    if this_host[13] == 4:
        x_0 = 20-1
        x_1 = 60-1
    if this_host[13] == 5:
        x_0 = 30-1
        x_1 = 70-1
    # print(train_init_goal.shape)
    # print(i)
    this_init_goal = test_init_goal[v_n*31 + f_n].reshape(8,)
    # if (init_goal[5] - local_y + 160)/320*54 < 54:
    this_init_goal = [(this_init_goal[0] - x_0)/40*10, (this_init_goal[1] - local_y + 160)/320*54,
                       this_init_goal[2],  this_init_goal[3],
                      (this_init_goal[4] - x_0)/40*10, (this_init_goal[5] - local_y + 160)/320*54,
                       this_init_goal[2],  this_init_goal[3]]
    x10f = []
    for k in range(10):
        x10f.append([(test_new_host[v_n][f_n+k][0] - x_0)/40*10,
                     (test_new_host[v_n][f_n+k][1] - local_y + 160)/320*54,
                      test_new_host[v_n][f_n+k][2],
                      test_new_host[v_n][f_n+k][3]])
    x10f = np.array(x10f)
    for k in np.arange(f_n+10, 40):
        x = np.vstack((x10f, [(test_new_host[v_n][k][0] - x_0)/40*10,
                              (test_new_host[v_n][k][1] - local_y + 160)/320*54,
                               test_new_host[v_n][k][2],
                               test_new_host[v_n][k][3]]))
        test_x.append(np.array(x).reshape(44,))
        test_i_g.append(this_init_goal)
        test_Occ.append(test_OccGrids[i])
    i += 1


test_Occ = np.array(test_Occ).astype(np.float32)
test_i_g = np.array(test_i_g).astype(np.float32)
test_x = np.array(test_x).astype(np.float32)


print("test occ shape:", test_Occ.shape)
print(test_i_g.shape)
print(test_x.shape)

test_Occ = np.array(test_Occ)
test_i_g = np.array(test_i_g)
test_x = np.array(test_x)

# train_Occ = train_Occ.reshape(len(train_Occ), 540)
# test_Occ = Variable(torch.from_numpy(test_Occ))
# test_i_g = Variable(torch.from_numpy(test_i_g))
# test_x = Variable(torch.from_numpy(test_x))
# cnn_train_data = CondDataset(train_Occ, train_init_goal, train_x, transform=transforms.ToTensor())

print("test occ shape:", test_Occ.shape)
print(test_i_g.shape)
print(test_x.shape)

cvae = CNNCVAE()

cvae.load_state_dict(torch.load('cnn-10f-cvae_params-epoch-700-vel1-traindata12-rl0818.pkl'))
# print(len(idx))
# print(idx)
# 40 42 2
for n in np.arange(1000, 1002, 2):
    final_sample_x = []
    final_traj = []
    test_case = 10*n
    test1 = test_Occ[test_case]
    m0 = np.loadtxt("jinghuai/OccGrids_3_7324-0.txt")
    m1 = np.loadtxt("jinghuai/OccGrids_3_7324-1.txt")
    m2 = np.loadtxt("jinghuai/OccGrids_3_7324-2.txt")
    m3 = np.loadtxt("jinghuai/OccGrids_3_7324-3.txt")
    m4 = np.loadtxt("jinghuai/OccGrids_3_7324-4.txt")
    m5 = np.loadtxt("jinghuai/OccGrids_3_7324-5.txt")
    m6 = np.loadtxt("jinghuai/OccGrids_3_7324-6.txt")
    m7 = np.loadtxt("jinghuai/OccGrids_3_7324-7.txt")
    m8 = np.loadtxt("jinghuai/OccGrids_3_7324-8.txt")
    m9 = np.loadtxt("jinghuai/OccGrids_3_7324-9.txt")
    test1 = [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9]
    test1 = np.array(test1)
    occ = []
    i_g = []
    v_n = int(n/31)
    f_n = n%31
    print(test_host[v_n*41+f_n])
    print(test_host[v_n * 41 + f_n + 9])
    print(test_i_g[test_case])
    print(test_new_host[v_n])
    for i in range(100):
        occ.append(test1)
        i_g.append(test_i_g[test_case])
    occ = Variable(torch.from_numpy(np.array(occ))).type(torch.FloatTensor)
    i_g = Variable(torch.from_numpy(np.array(i_g))).type(torch.FloatTensor)
    print(occ.shape)
    print(i_g.shape)
    recon_x = cvae.inference(occ, i_g, 100)
    print(recon_x)
    # print(recon_x)
    # print(test_x[0:40])
    for f in range(11):
        if f!= 10:
            plt.figure(figsize=(6,1.35))
            row = 0
            this_occ = occ[0][f].numpy()
            for i in np.arange(10):
                col = 0
                for j in np.arange(54):
                    if this_occ[row][col] == 1:
                        plt.scatter(j, i, color="red", s=50, marker=",")
                    elif this_occ[row][col] == 0:
                       plt.scatter(j, i, color="green", s=50, marker=",")
                    col += 1
                row += 1
            for i in range(100):
                if i%10==0:
                # plt.scatter(recon_x.data.numpy()[i][f*4], recon_x.data.numpy()[i][f*4+1], color="blue", s=5)
                    plt.scatter(recon_x.data.numpy()[i][f * 4 + 1], recon_x.data.numpy()[i][f * 4]+(2*random.random()-1)/2, color="blue", s=3)
                    final_sample_x.append([recon_x.data.numpy()[i][f * 4 + 1], recon_x.data.numpy()[i][f * 4]+(2*random.random()-1)/2])
            final_traj = []
            for i in range(10):
               # plt.scatter(test_x[test_case][i * 4], test_x[test_case][i * 4 + 1], color="black",s=5)
                # plt.scatter(test_x[test_case][i * 4 + 1], test_x[test_case][i * 4], color="black")
                final_traj.append([test_x[test_case][i * 4 + 1], test_x[test_case][i * 4]])
            ax = plt.gca()
            final_traj = np.array(final_traj)
            plt.plot(final_traj[:, 0], final_traj[:, 1], color="black")
            ax.set_aspect(1)
            # plt.xlim(5, 47)
            plt.axis('off')
            plt.show()
        else:
            plt.figure(figsize=(6, 1.35))
            row = 0
            this_occ = occ[0][f-1].numpy()
            for i in np.arange(10):
                col = 0
                for j in np.arange(54):
                    if this_occ[row][col] == 1:
                        # plt.scatter(i, j, color="red")
                        plt.scatter(j, i, color="red", s=50, marker=",")
                    elif this_occ[row][col] == 0:
                        # plt.scatter(i, j, color="green")
                        plt.scatter(j, i, color="green", s=50, marker=",")
                    col += 1
                row += 1
            for i in range(100):
                # plt.scatter(recon_x.data.numpy()[i][f*4], recon_x.data.numpy()[i][f*4+1], color="orange")
                plt.scatter(recon_x.data.numpy()[i][f * 4 + 1], recon_x.data.numpy()[i][f * 4] + (2 * random.random() - 1)/2, color="orange", marker="^", s=3)

            ax = plt.gca()
            ax.set_aspect(1)
            # plt.xlim(5, 47)
            plt.axis('off')
            plt.show()
    f = 0
    row = 0
    this_occ = occ[0][f].numpy()
    plt.figure(figsize=(6, 1.35))
    for i in np.arange(10):
        col = 0
        for j in np.arange(54):
            if this_occ[row][col] == 1:
                # plt.scatter(i, j, color="red")
                plt.scatter(j, i, color="red", s=50, marker=",")
            elif this_occ[row][col] == 0:
                # plt.scatter(i, j, color="green")
                plt.scatter(j, i, color="green", s=50, marker=",")
            col += 1
        row += 1
    f = 10
    for i in range(100):
        # plt.scatter(recon_x.data.numpy()[i][f*4], recon_x.data.numpy()[i][f*4+1], color="orange")
        plt.scatter(recon_x.data.numpy()[i][f * 4 + 1],
                    recon_x.data.numpy()[i][f * 4] + (2 * random.random() - 1)/2, color="orange", marker="^", s=3)
    final_sample_x = np.array(final_sample_x)
    final_traj = np.array(final_traj)
    plt.scatter(final_sample_x[:,0], final_sample_x[:,1], color="blue", s=3)
    plt.plot(final_traj[:,0], final_traj[:,1], color="black")
    ax = plt.gca()
    ax.set_aspect(1)
    # plt.xlim(5, 47)
    plt.axis('off')
    plt.show()



