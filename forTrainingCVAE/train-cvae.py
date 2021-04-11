import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


OccGrids_1 = []
for i in range(488*31):
    if i%1000 == 0:
        print(i)
    Occ = []
    for j in range(10):
        Occ.append(np.loadtxt("../OccGrids_1_10f/OccGrids_1_"+str(i)+"-"+str(j)+".txt"))
    OccGrids_1.append(Occ)
OccGrids_1 = np.array(OccGrids_1)
OccGrids_1 = OccGrids_1.reshape(488*31, 10, 10, 54)
print(OccGrids_1.shape)


OccGrids_2 = []
for i in range(382*31):
    if i%1000 == 0:
        print(i)
    Occ = []
    for j in range(10):
        Occ.append(np.loadtxt("../OccGrids_2_10f/OccGrids_2_" + str(i) + "-" + str(j) + ".txt"))
    OccGrids_2.append(Occ)
OccGrids_2 = np.array(OccGrids_2)
OccGrids_2 = OccGrids_2.reshape(382*31, 10, 10, 54)
print(OccGrids_2.shape)

OccGrids_3 = []
for i in range(404*31):
    if i%1000 == 0:
        print(i)
    Occ = []
    for j in range(10):
        Occ.append(np.loadtxt("../OccGrids_3_10f/OccGrids_3_" + str(i) + "-" + str(j) + ".txt"))
    OccGrids_3.append(Occ)
OccGrids_3 = np.array(OccGrids_3)
OccGrids_3 = OccGrids_3.reshape(404*31, 10, 10, 54)
print(OccGrids_3.shape)

change_lane_label_1 = np.loadtxt("../change_lane_label_1.txt")
new_host_1 = np.loadtxt("../new_host_1.txt")
host_1 = np.loadtxt("../host_1.txt")
init_goal_1 = np.loadtxt("../init_goal_1.txt")
init_goal_1 = np.array(init_goal_1).reshape(488*31,2,4)
print(init_goal_1.shape)
new_host_1 = np.array(new_host_1).reshape(488,40,4)
print(new_host_1.shape)

change_lane_label_2 = np.loadtxt("../change_lane_label_2.txt")
new_host_2 = np.loadtxt("../new_host_2.txt")
host_2 = np.loadtxt("../host_2.txt")
init_goal_2 = np.loadtxt("../init_goal_2.txt")
init_goal_2 = np.array(init_goal_2).reshape(382*31,2,4)
print(init_goal_2.shape)
new_host_2 = np.array(new_host_2).reshape(382,40,4)
print(new_host_2.shape)

change_lane_label_3 = np.loadtxt("../change_lane_label_3.txt")
new_host_3 = np.loadtxt("../new_host_3.txt")
host_3 = np.loadtxt("../host_3.txt")
init_goal_3 = np.loadtxt("../init_goal_3.txt")
init_goal_3 = np.array(init_goal_3).reshape(404*31,2,4)
print(init_goal_3.shape)
new_host_3 = np.array(new_host_3).reshape(404,40,4)
print(new_host_3.shape)

train_OccGrids = np.concatenate((OccGrids_1, OccGrids_2))
test_OccGrids = OccGrids_3
train_change_lane_label = np.concatenate((change_lane_label_1, change_lane_label_2))
test_change_lane_label = change_lane_label_3
train_host = np.concatenate((host_1, host_2))
test_host = host_3
train_new_host = np.concatenate((new_host_1, new_host_2))
test_new_host = new_host_3
train_init_goal = np.concatenate((init_goal_1, init_goal_2))
test_init_goal = init_goal_3
print("train:")
print(train_OccGrids.shape)
print(train_change_lane_label.shape)
print(train_host.shape)
print(train_new_host.shape)
print(train_init_goal.shape)
print("test:")
print(test_OccGrids.shape)
print(test_change_lane_label.shape)
print(test_host.shape)
print(test_new_host.shape)
print(test_init_goal.shape)


class CondDataset(torch.utils.data.Dataset):
    def __init__(self, occ_grid, init_goal, x, transform=None):
        self.data = occ_grid
        self.init_goal = init_goal
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
        return img, self.init_goal[index], self.x[index]

    def __len__(self):
        return len(self.data)

import random

train_Occ = []
train_i_g = []
train_x = []
test_Occ = []
test_i_g = []
test_x = []
# train_size = 30000
# test_size = 39494 - 30000
# test = random.sample(range(train_size + test_size), test_size)
# np.savetxt("test.txt", test)


i = 0
for v in train_OccGrids:
    if i%1000 == 0:
        print(i)
    v_n = int(i/31)
    f_n = i%31
    # if change_lane_label[v_n] == 1:
    local_y = train_new_host[v_n][f_n][1]
    this_host = train_host[v_n*41 + f_n]
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
    this_init_goal = train_init_goal[i].reshape(8,)
    # if (init_goal[5] - local_y + 160)/320*54 < 54:
    this_init_goal = [(this_init_goal[0] - x_0)/40*10, (this_init_goal[1] - local_y + 160)/320*54,
                       this_init_goal[2],  this_init_goal[3],
                      (this_init_goal[4] - x_0)/40*10, (this_init_goal[5] - local_y + 160)/320*54,
                       this_init_goal[2],  this_init_goal[3]]
    x10f = []
    for k in range(10):
        x10f.append([(train_new_host[v_n][f_n + k][0] - x_0) / 40 * 10,
                     (train_new_host[v_n][f_n + k][1] - local_y + 160) / 320 * 54,
                     train_new_host[v_n][f_n + k][2],
                     train_new_host[v_n][f_n + k][3]])
    x10f = np.array(x10f)
    for k in np.arange(f_n + 10, 40):
        x = np.vstack((x10f, [(train_new_host[v_n][k][0] - x_0) / 40 * 10,
                              (train_new_host[v_n][k][1] - local_y + 160) / 320 * 54,
                              train_new_host[v_n][k][2],
                              train_new_host[v_n][k][3]]))
        train_x.append(np.array(x).reshape(44, ))
        train_i_g.append(this_init_goal)
        train_Occ.append(train_OccGrids[i])
    i += 1

i = 0
for v in test_OccGrids:
    if i%1000 == 0:
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
    this_init_goal = test_init_goal[i].reshape(8,)
    # if (init_goal[5] - local_y + 160)/320*54 < 54:
    this_init_goal = [(this_init_goal[0] - x_0)/40*10, (this_init_goal[1] - local_y + 160)/320*54,
                       this_init_goal[2],  this_init_goal[3],
                      (this_init_goal[4] - x_0)/40*10, (this_init_goal[5] - local_y + 160)/320*54,
                       this_init_goal[2],  this_init_goal[3]]
    x10f = []
    for k in range(10):
        x10f.append([(test_new_host[v_n][f_n + k][0] - x_0) / 40 * 10,
                     (test_new_host[v_n][f_n + k][1] - local_y + 160) / 320 * 54,
                     test_new_host[v_n][f_n + k][2],
                     test_new_host[v_n][f_n + k][3]])
    x10f = np.array(x10f)
    for k in np.arange(f_n + 10, 40):
        x = np.vstack((x10f, [(test_new_host[v_n][k][0] - x_0) / 40 * 10,
                              (test_new_host[v_n][k][1] - local_y + 160) / 320 * 54,
                              test_new_host[v_n][k][2],
                              test_new_host[v_n][k][3]]))
        test_x.append(np.array(x).reshape(44, ))
        test_i_g.append(this_init_goal)
        test_Occ.append(test_OccGrids[i])
    i += 1


train_Occ = np.array(train_Occ).astype(np.float32)
train_i_g = np.array(train_i_g).astype(np.float32)
train_x = np.array(train_x).astype(np.float32)
test_Occ = np.array(test_Occ).astype(np.float32)
test_i_g = np.array(test_i_g).astype(np.float32)
test_x = np.array(test_x).astype(np.float32)

print("train occ shape:", train_Occ.shape)
print(train_i_g.shape)
print(train_x.shape)
print("test occ shape:", test_Occ.shape)
print(test_i_g.shape)
print(test_x.shape)


# train_Occ = train_Occ[0:90000]
# train_init_goal = train_init_goal[0:90000]
# train_x = train_x[0:90000]

EPOCH = 30              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 256
LR = 0.0001               # learning rate

train_Occ = train_Occ.reshape(len(train_Occ), 10, 10, 54)
test_Occ = Variable(torch.from_numpy(np.array(test_Occ).reshape(len(test_Occ), 10, 10, 54).astype(np.float32)))
test_i_g = Variable(torch.from_numpy(np.array(test_i_g).astype(np.float32)))
test_x = Variable(torch.from_numpy(np.array(test_x).astype(np.float32)))
# cnn_train_data = CondDataset(train_Occ, train_init_goal, train_x, transform=transforms.ToTensor())
train_data = CondDataset(train_Occ, train_i_g, train_x)

print(train_data.data.shape)
print(train_data.init_goal.shape)
print(train_data.x.shape)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=10,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  
            ),  
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  
        )
        self.conv2 = nn.Sequential( 
            # nn.Conv2d(20, 16, 5, 27),
            nn.Conv2d(16, 32, 5, 1, 2), 
            nn.ReLU(),  # activation
            nn.MaxPool2d(2), 
        )
        self.cnnout = nn.Linear(832, 128) 

    def forward(self, c):
        # c = c.type(torch.FloatTensor)
        c = self.conv1(c)
        c = self.conv2(c)
        c = c.view(c.size(0), -1) 
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


cvae = CNNCVAE()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_func(z_mu, z_logvar, b_x, recon_x):
    KL_loss = 10**-3 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
    MSE_loss = torch.sqrt(torch.sum((b_x - recon_x) ** 2))/ BATCH_SIZE
    return torch.mean(KL_loss + MSE_loss)

# cvae = cvae.cuda()
optimizer = torch.optim.Adam(cvae.parameters(), lr=LR)   # optimize all cvae parameters
losses = []
t_losses = []

# training and testing
for epoch in range(EPOCH):
    for step, (b_c, b_i_g, b_x) in enumerate(train_loader):        # gives batch data
        # print(step)
        # print(b_x.shape)
        # print(b_i_g.shape)
        # print(b_c.shape)
        recon_x, z_mu, z_logvar, z = cvae(b_x, b_c, b_i_g)                               # output
        n = random.randint(0, len(test_x) - BATCH_SIZE)
        t_x = test_x[n:n + BATCH_SIZE]
        t_c = test_Occ[n:n + BATCH_SIZE]
        t_i_g = test_i_g[n:n + BATCH_SIZE]
        t_recon_x, t_z_mu, t_z_logvar, t_z = cvae(t_x, t_c, t_i_g)
        # print(recon_x)
        # print(z_mu.shape)
        # print(z_logvar.shape)
        # print(z)
        # KL_loss = 10**-4 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        # print(KL_loss)
        # MSE_loss = torch.sqrt(torch.sum(weight * (b_x - recon_x) ** 2))
        # print(MSE_loss)
        loss = loss_func(z_mu, z_logvar, b_x, recon_x)
        t_loss = loss_func(t_z_mu, t_z_logvar, t_x, t_recon_x)
        # print(type(loss))
        # break
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            # print(test_x.shape)
            # test_x = test_x.reshape(1900,1,10,54)
            # test_output = cnn(test_x)                   # (samples, time_step, input_size)
            # pred_y = torch.max(test_output, 1)[1].data.numpy()
            # label_y = torch.max(test_output, 1)[1].data.numpy()
            # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            # accuracy_label = float((label_y == labels_check).astype(int).sum()) / float(test_y.size)
            # print(label_y)
            # print(labels_check)
            # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test loss: %.4f' % t_loss.data.numpy())
        losses.append(loss.data.numpy())
        t_losses.append(t_loss.data.numpy())
        # print(losses)
        # break
    # break

torch.save(cvae.state_dict(), 'cvae_params-epoch-30-vel1-traindata12-rl.pkl')
losses = np.array(losses)
np.savetxt("losses-epoch-30-vel1-traindata12-rl.txt", losses)
t_losses = np.array(t_losses)
np.savetxt("test-losses-epoch-30-vel1-traindata12-rl.txt", t_losses)
