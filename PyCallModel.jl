const Vec2f = SVector{2, Float64}
const Vec4f = SVector{4, Float64}
const VecFrame = SVector{18, Float64}
using LinearAlgebra
using StaticArrays
using Clustering

using PyCall
using Conda
using Pkg

ENV["PYTHON"] = "D:\\Anaconda\\python.exe"
Pkg.build("PyCall")
println(PyCall.conda)
println(PyCall.libpython)
pyimport_conda("torch", "pytorch", "pytorch")
pyimport_conda("torchvision", "torchvision", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/")

py"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.cluster import KMeans

class Rectangle:
    def __init__(self, center, width, height):
        self.V1 = np.add(center, [-width/2, -height/2])
        self.V2 = np.add(center, [width/2, -height/2])
        self.V3 = np.add(center, [width/2, height/2])
        self.V4 = np.add(center, [-width/2, height/2])
        self.V = np.array([self.V1, self.V2, self.V3, self.V4])
        self.N = 4

    def isInside(self, q):
        # note: n'x = n'b
        q_vec = [q[0], q[1]]
        for n in range(self.N):
            p1 = self.V[n]
            if n < self.N - 1:
                p2 = self.V[n+1]
            else:
                p2 = self.V[0]
            u = np.array([p2[0]-p1[0], p2[1]-p1[1]]) #p2-p1
            norm = np.dot([[0, 1], [-1,0]], u) # normla vector
            px = np.array([q[0]-p1[0], q[1]-p1[1]]) # x-p1
            if np.dot(px.conj().T, norm) > 0:
                return False
        return True

    def isIntersect(self, q1_, q2_):
        q1 = q1_[0:2]
        q2 = q2_[0:2]
        return self._isIntersect(q1, q2)

    def _isIntersect(self, q1, q2):
        for n in range(self.N):
            p1 = self.V[n]
            if n < self.N-1:
                p2 = self.V[n+1]
            else:
                p2 = self.V[1]
            if self.isIntersect_4(p1, p2, q1, q2):
                return True
        return False

    def isInside_seq(self, q_seq):
        for q in q_seq:
            if self.isInside(q):
                return True
        return False

    def isIntersect_4(self, p1, p2, q1, q2):
        # solve [p2-p1, q2-q1]*[s; t]=[q1-p1] or A*[s; t]=B
        # for computational efficiency, not using inv() func
        # A = [a, b; c, d]
        # B = [e; f]
        a = p2[0]-p1[0]
        b = -(q2[0]-q1[0])
        c = p2[1]-p1[1]
        d = -(q2[1]-q1[1])
        e = q1[0]-p1[0]
        f = q1[1]-p1[1]
        det = a*d-b*c
        if abs(det) < 1e-5:
            return False

        s = 1/det*(d*e-b*f)
        t = 1/det*(-c*e+a*f)
        return ((0.0<=s<=1.0) and (0.0<=t<=1.0))

class World:
    def __init__(self, x_min, x_max, v_min, v_max, Pset, vehicle, delta=0.0):
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max
        self.Pset = Pset
        self.vehicle = vehicle
        self.delta = delta

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def isValid(self, s_q):
        # check if the sampled point is inside the world"
        vec = np.array([s_q[2], s_q[3]])
        if s_q[2] == 0:
            s_q[2] = 1e-20
        angel = np.rad2deg(np.arctan(s_q[3] / s_q[2]))
        angle_orth1 = np.deg2rad(angel - 90)
        angle_orth2 = np.deg2rad(angel + 90)
        norm_vec = self.normalize(vec)
        norm_vec_orth1 = np.array([np.cos(angle_orth1), np.sin(angle_orth1)])
        norm_vec_orth2 = np.array([np.cos(angle_orth2), np.sin(angle_orth2)])
        front_edge_center = np.add([s_q[0], s_q[1]], (self.vehicle[1] / 2) * norm_vec)
        rear_edge_center = np.add([s_q[0], s_q[1]], (self.vehicle[1] / 2) * -norm_vec)
        v1 = np.add(front_edge_center, self.vehicle[0] / 2 * norm_vec_orth1)
        v2 = np.add(front_edge_center, self.vehicle[0] / 2 * norm_vec_orth2)
        v3 = np.add(rear_edge_center, self.vehicle[0] / 2 * norm_vec_orth1)
        v4 = np.add(rear_edge_center, self.vehicle[0] / 2 * norm_vec_orth2)
        vertex = np.array([v1, v2, v4, v3])
        for p in vertex:
            if not self.x_min[0] < p[0] < self.x_max[0]:
                # print(s_q)
                # print(p)
                # print("not self.x_min[0]< p[0] <self.x_max[0]")
                return False
            if not self.x_min[1] < p[1] < self.x_max[1]:
                # print(s_q)
                # print("not self.x_min[1]< p[1] <self.x_max[1]")
                return False
        if not self.v_min[0] < s_q[2] and s_q[2] < self.v_max[0]:
            # print(s_q)
            # print("not self.v_min[0]<s_q[2] and s_q[2]<self.v_max[0]")
            return False
        if not self.v_min[1] < s_q[2] and s_q[3] < self.v_max[1]:
            # print(s_q)
            # print("not self.v_min[1]<s_q[2] and s_q[3]<self.v_max[1]")
            return False

        if self.isIntersect(vertex[0], vertex[1]):
            # print(s_q)
            # print("self.isIntersect(vertex[0], vertex[1])")
            return False
        if self.isIntersect(vertex[1], vertex[2]):
            # print(s_q)
            # print("self.isIntersect(vertex[1], vertex[2])")
            return False
        if self.isIntersect(vertex[2], vertex[3]):
            # print(s_q)
            # print("self.isIntersect(vertex[2], vertex[3])")
            return False
        if self.isIntersect(vertex[3], vertex[0]):
            # print(s_q)
            # print("self.isIntersect(vertex[3], vertex[0])")
            return False
        for v in vertex:
            if self.isInside(v):
                # print(s_q)
                # print("self.isInside(v)")
                return False
        return True

    def isValid_pos(self, s_q):
        # check if the sampled point is inside the world"
        # print("world isValid self.x_min[0]:", self.x_min[0])
        # print("world isValid s_q[0]:", s_q[0])
        if not self.x_min[0] < s_q[0] and s_q[0] < self.x_max[0]:
            return False
        if not self.x_min[1] < s_q[1] and s_q[1] < self.x_max[1]:
            return False
        for P in self.Pset:
            if P.isInside(s_q[0:2]):
                return False
        return True

    def isVehicle(self, s_q):
        if self.vehicle.isInside(s_q[0:2]):
            return True
        return False

    def isValid_seq(self, q_set):
        # check validity for multiple points.
        # will be used for piecewize path consited of multiple points
        for q in q_set:
            if not self.isValid(q):
                return False
        return True

    def isIntersect(self, q1, q2):
        for P in self.Pset:
            if P.isIntersect(q1, q2):
                return True
        return False

    def isInside(self, q):
        for P in self.Pset:
            if P.isInside(q):
                return True
        return False

def distance(x, y):
    return np.sqrt( (x[1]-y[1])**2 + (x[0]-y[0])**2 )


def generate_occ_grid(host_i, neighbors, r, left_edge=0, right_edge=60):
    lane1_c = 7.231027419583393
    lane2_c = 18.84918099161813
    lane3_c = 29.990037979406054
    lane4_c = 40.99582664476257
    lane5_c = 53.099915981198585
    lane_c = [7.231027419583393, 18.84918099161813, 29.990037979406054, 40.99582664476257, 53.099915981198585]

    center = [host_i[0], host_i[1]]
    lower = center[1] - 150 - 10
    upper = center[1] + 150 + 10

    if host_i[2] == 1:
        x_0 = 0 - 1
        x_1 = 40 - 1
    if host_i[2] == 2:
        x_0 = 0 - 1
        x_1 = 40 - 1
    if host_i[2] == 3:
        x_0 = 10 - 1
        x_1 = 50 - 1
    if host_i[2] == 4:
        x_0 = 20 - 1
        x_1 = 60 - 1
    if host_i[2] == 5:
        x_0 = 30 - 1
        x_1 = 70 - 1

    x_min = [x_0, lower]
    x_max = [x_1, upper]
    v_min = [-10, -10]
    v_max = [10, 100]

    vehicle_i = Rectangle((host_i[0], host_i[1]), host_i[3], host_i[4])

    final_Occs = []

    for frame in neighbors:
        Pset = []
        for v in frame:
            Pset.append(Rectangle((v[4], v[5]), v[9], v[8]))
        W = World(x_min, x_max, v_min, v_max, Pset, vehicle_i, delta=0.0)
        OccGrid = []  # 1 -- occupied  0 -- free

        row = 0

        for i in np.arange(x_min[0] + 1, x_max[0], 4):
            col = 0
            row += 1
            for j in np.arange(x_min[1] + 1, x_max[1], 6):
                col += 1
                if not W.isValid_pos([i, j, 0, 20]):
                    OccGrid.append(1)
                else:
                    OccGrid.append(0)
        OccGrid = np.array(OccGrid).reshape(row, col)
        final_Occs.append(OccGrid)
    final_Occs = np.array(final_Occs).reshape(10, row, col)
    return final_Occs, center, vehicle_i, x_min, x_max

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

def find_neighbors(x, lane_id, vehicles, r):
    neighbors = [[]]
    for v in vehicles:
        if distance(x, [v[4], v[5]]) < r and (v[13]==lane_id or v[13]==lane_id-1 or v[13]==lane_id-1) and v[13] < 6:
            neighbors.append(v)
    del neighbors[0]
    return neighbors

def sample_points(x, goal, lane_id, vehicles, r, n_samples=1000):
    # print("start CVAE sampling")
    neighbors = []
    for frame in vehicles:
        neighbors.append(find_neighbors(x, lane_id, frame, r))
    width = 7
    length = 16
    host = [x[0], x[1], lane_id, width, length]
    # print("neighbors:", len(neighbors))
    OccGrids, c, v, x_min, x_max = generate_occ_grid(host, neighbors, 150)
    OccGrids = np.array(OccGrids).astype(np.float32)
    # i = 0
    # for Occ in OccGrids:
    #     np.savetxt("./time_lag_exp/OccGrids3_"+str(i)+".txt", Occ)
    #     i += 1
    local_y = x[1]
    if lane_id == 1:
        x_0 = 0-1
        x_1 = 40-1
    if lane_id == 2:
        x_0 = 0-1
        x_1 = 40-1
    if lane_id == 3:
        x_0 = 10-1
        x_1 = 50-1
    if lane_id == 4:
        x_0 = 20-1
        x_1 = 60-1
    if lane_id == 5:
        x_0 = 30-1
        x_1 = 70-1
    # print(x)
    # print(goal)
    init_goal = [(x[0] - x_0)/40*10, (x[1] - local_y + 160)/320*54,
                  x[2],  x[3],
                 (goal[0] - x_0)/40*10, (goal[1] - local_y + 160)/320*54,
                  goal[2],  goal[3]]
    # print(init_goal)
    init_goal = np.array(init_goal).astype(np.float32)
    test_Occ = []
    test_init_goal = []
    for i in range(n_samples):
        test_Occ.append(OccGrids)
        test_init_goal.append(init_goal)
    test_Occ = np.array(test_Occ)
    test_init_goal = np.array(test_init_goal)
    test_Occ = Variable(torch.from_numpy(test_Occ.astype(np.float32)))
    test_init_goal = Variable(torch.from_numpy(test_init_goal.astype(np.float32)))
    # print(test_Occ.shape)
    # print(test_init_goal.shape)
    cvae = CNNCVAE()
    cvae.load_state_dict(torch.load('cnn-10f-cvae_params-epoch-700-vel1-traindata12-rl0818.pkl'))
    print("finish load model")
    recon_x = cvae.inference(test_Occ, test_init_goal, n_samples)
    print(recon_x.shape)
    Pset = []
    for x in recon_x:
        for i in range(11):
            Pset.append([x[i*4]/10*40+x_0, x[i*4 + 1]/54*320+local_y-160,
                    x[i*4 + 2], x[i*4 + 3]])
    Pset = np.array(Pset)
    print(Pset.shape)
    return Pset
"""

function sample(init::Vec4f, goal::Vec4f, lane_id::Int64, frames, n_samples::Int64, r::Float64, world)
    Pset = py"sample_points"(init, goal, lane_id, frames, 150.0, n_samples*10)
    i = 1
    res = Vec4f[]
    n_samples = n_samples * 10
    while i < n_samples
        p = Vec4f(convert(Float64, Pset[i]),
                  convert(Float64, Pset[1*n_samples*11+i]),
                  convert(Float64, Pset[2*n_samples*11+i]),
                  convert(Float64, Pset[3*n_samples*11+i]))
        push!(res, p)
        i += 1
    end
    return res
end
