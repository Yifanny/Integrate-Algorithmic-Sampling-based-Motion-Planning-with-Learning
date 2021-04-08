import numpy as np

file1 = open("us101/trajectories-0750am-0805am.txt", "r")
file2 = open("us101/trajectories-0805am-0820am.txt", "r")
file3 = open("us101/trajectories-0820am-0835am.txt", "r")

"""
0: Vehicle_ID
1: Frame_ID
2: Total_Frames
3: Global_Time
4: Local_X -- Lateral (X) coordinate of the front center of the vehicle in feet with respect to the left-most edge of the section in the direction of travel.
5: Local_Y -- Longitudinal (Y) coordinate of the front center of the vehicle in feet with respect to the entry edge of the section in the direction of travel.
6: Global_X -- X Coordinate of the front center of the vehicle in feet based on CA State Plane III in NAD83. Attribute Domain Val
7: Global_Y
8: v_length -- Length of vehicle in feet
9: v_Width -- Width of vehicle in feet
10: v_Class -- Vehicle type: 1 - motorcycle, 2 - auto, 3 - truck
11: v_Vel -- Instantaneous velocity of vehicle in feet/second.
12: v_Acc -- Instantaneous acceleration of vehicle in feet/second square.
13: Lane_ID -- Current lane position of vehicle.
            Lane 1 is farthest left lane; 
            lane 5 is farthest right lane. 
            Lane 6 is the auxiliary lane between Ventura Boulevard on-ramp and the Cahuenga Boulevard off-ramp. 
            Lane 7 is the on-ramp at Ventura Boulevard, 
            and Lane 8 is the off-ramp at Cahuenga Boulevard.
14: Preceding -- Vehicle ID of the lead vehicle in the same lane.
                A value of '0' represents no preceding vehicle - occurs at the end of the study section and off-ramp due to the fact that only complete trajectories were recorded by this data collection effort (vehicles already in the section at the start of the study period were not recorded).
15: Following -- Vehicle ID of the vehicle following the subject vehicle in the same lane.
                A value of '0' represents no following vehicle - occurs at the beginning of the study section and onramp due to the fact that only complete trajectories were recorded by this data collection effort (vehicle that did not traverse the downstream boundaries of the section by the end of the study period were not recorded).
16: Space_Headway -- Space Headway in feet. Spacing provides the distance between the frontcenter of a vehicle to the front-center of the preceding vehicle.
17: Time_Headway -- Time Headway in seconds.
                    Time Headway provides the time to travel from the front-center of a vehicle (at the speed of the vehicle) to the front-center of the preceding vehicle.
                    A headway value of 99.
"""
data_us101_1 = []
data_us101_2 = []
data_us101_3 = []
for row in file1.readlines():
    row = row.split(' ')
    while True:
        try:
            row.remove('')
        except ValueError:
            break
    row.remove('\n')
    row = tuple(row)
    data_us101_1.append(row)
for row in file2.readlines():
    row = row.split(' ')
    while True:
        try:
            row.remove('')
        except ValueError:
            break
    row.remove('\n')
    row = tuple(row)
    data_us101_2.append(row)
for row in file3.readlines():
    row = row.split(' ')
    while True:
        try:
            row.remove('')
        except ValueError:
            break
    row.remove('\n')
    row = tuple(row)
    data_us101_3.append(row)
print(len(data_us101_1))
print(len(data_us101_2))
print(len(data_us101_3))

dtype =  [('Vehicle_ID', np.int64), ('Frame_ID', np.int64), ('Total_Frames', np.int64), ('Global_Time', np.int64), ('Local_X', float), ('Local_Y', float),
          ('Global_X', float), ('Global_Y', float), ('v_length', float), ('v_width', float), ('v_Class', np.int64), ('v_Vel', float), ('v_Acc', float),
          ('Lane_ID', np.int64), ('Preceding', np.int64), ('Following',np.int64), ('Space_Headway', float), ('Time_Headway', float)]
# print(data_us101[0])
data_us101_1 = np.array(data_us101_1, dtype=dtype)
data_us101_2 = np.array(data_us101_2, dtype=dtype)
data_us101_3 = np.array(data_us101_3, dtype=dtype)
# data_us101 = np.array(data_us101, dtype=dtype)
# data_i80 = np.array(data_i80, dtype=dtype)
# data_lankershim = np.array(data_lankershim, dtype=dtype)

data_us101_sorted_1 = np.sort(data_us101_1, order=['Frame_ID', 'Vehicle_ID', 'Total_Frames'])
data_us101_sorted_2 = np.sort(data_us101_2, order=['Frame_ID', 'Vehicle_ID', 'Total_Frames'])
data_us101_sorted_3 = np.sort(data_us101_3, order=['Frame_ID', 'Vehicle_ID', 'Total_Frames'])

data_us101_frame_1 = []
frame = []
for i in range(len(data_us101_sorted_1)):
    if i % 10000 == 0:
        print(i)
    if i == 0:
        frame.append(data_us101_sorted_1[i])
        continue
    if data_us101_sorted_1[i][1] == data_us101_sorted_1[i - 1][1]:
        frame.append(data_us101_sorted_1[i])
    else:
        data_us101_frame_1.append(frame)
        frame = []
        frame.append(data_us101_sorted_1[i])
        if i == len(data_us101_sorted_1) - 1:
            data_us101_frame_1.append(frame)

data_us101_frame_2 = []
frame = []
for i in range(len(data_us101_sorted_2)):
    if i % 10000 == 0:
        print(i)
    if i == 0:
        frame.append(data_us101_sorted_2[i])
        continue
    if data_us101_sorted_2[i][1] == data_us101_sorted_2[i - 1][1]:
        frame.append(data_us101_sorted_2[i])
    else:
        data_us101_frame_2.append(frame)
        frame = []
        frame.append(data_us101_sorted_2[i])
        if i == len(data_us101_sorted_2) - 1:
            data_us101_frame_2.append(frame)

data_us101_frame_3 = []
frame = []
for i in range(len(data_us101_sorted_3)):
    if i % 10000 == 0:
        print(i)
    if i == 0:
        frame.append(data_us101_sorted_3[i])
        continue
    if data_us101_sorted_3[i][1] == data_us101_sorted_3[i - 1][1]:
        frame.append(data_us101_sorted_3[i])
    else:
        data_us101_frame_3.append(frame)
        frame = []
        frame.append(data_us101_sorted_3[i])
        if i == len(data_us101_sorted_3) - 1:
            data_us101_frame_3.append(frame)

print('finish loop')

data_us101_v_sorted_1 = np.sort(data_us101_1, order=['Vehicle_ID', 'Frame_ID'])
data_us101_v_sorted_2 = np.sort(data_us101_2, order=['Vehicle_ID', 'Frame_ID'])
data_us101_v_sorted_3 = np.sort(data_us101_3, order=['Vehicle_ID', 'Frame_ID'])

import matplotlib.pyplot as plt


def distance(x, y):
    return np.sqrt( (x[1]-y[1])**2 + (x[0]-y[0])**2 )


def get_change_lane_list(data_us101_v_sorted, data_us101_frame, offset):
    vehicles_change_lane_list = []
    change_lane_label = []
    i = 0
    print(len(data_us101_v_sorted))
    for i in range(len(data_us101_v_sorted)):
        if i == 0:
            continue
        if data_us101_v_sorted[i][13] > 5 or data_us101_v_sorted[i-1][13] > 5:
            continue
        if data_us101_v_sorted[i][13] != data_us101_v_sorted[i-1][13] and data_us101_v_sorted[i][0] == data_us101_v_sorted[i-1][0] and data_us101_v_sorted[i][2] == data_us101_v_sorted[i-1][2]:
            if (data_us101_v_sorted[i][0], data_us101_v_sorted[i][2], data_us101_v_sorted[i][1]) not in vehicles_change_lane_list:
                ok = True
                thereisfrontvehicle = True
                lane_id = data_us101_v_sorted[i-1][13]
                next_lane_id = data_us101_v_sorted[i][13]
                v_id = data_us101_v_sorted[i-1][0]
                for j in range(20):
                    if data_us101_v_sorted[i-j-1][13] != lane_id:
                        ok = False
                        break
                    if v_id != data_us101_v_sorted[i-j-1][0] or v_id != data_us101_v_sorted[i+j][0]:
                        ok = False
                        break
                    if data_us101_v_sorted[i+j][13] != next_lane_id:
                        ok = False
                        break
                for j in range(20):
                    center = [data_us101_v_sorted[i-j-1][4], data_us101_v_sorted[i-j-1][5]]
                    lane_id = data_us101_v_sorted[i-j-1][13]
                    frame_id = data_us101_v_sorted[i-j-1][1] - offset
                    frame = data_us101_frame[frame_id]
                    front_vehicle = 0
                    back_vehicle = 0
                    for v in frame:
                        if v[5] > center[1] and (v[13] == lane_id or v[13] == lane_id-1 or v[13] == lane_id+1) and distance(center, [v[4], v[5]]) < 1000:
                            front_vehicle += 1
                        if v[5] < center[1] and (v[13] == lane_id or v[13] == lane_id-1 or v[13] == lane_id+1) and distance(center, [v[4], v[5]]) < 1000:
                            back_vehicle += 1
                    if front_vehicle > 2 and back_vehicle > 2:
                        thereisfrontvehicle = (thereisfrontvehicle and True)
                    else:
                        thereisfrontvehicle = (thereisfrontvehicle and False)
                if ok and thereisfrontvehicle:
                    vehicles_change_lane_list.append((data_us101_v_sorted[i-1][0], data_us101_v_sorted[i-1][2], data_us101_v_sorted[i-1][1]))
                    if data_us101_v_sorted[i][13] > data_us101_v_sorted[i-1][13]:
                        change_lane_label.append(3) # right
                    else:
                        change_lane_label.append(1) # left
        if i%100000 == 0:
            print(i)
    return vehicles_change_lane_list, change_lane_label

vehicles_change_lane_list_1, change_lane_label_1 = get_change_lane_list(data_us101_v_sorted_1, data_us101_frame_1, data_us101_frame_1[0][0][1])
vehicles_change_lane_list_2, change_lane_label_2 = get_change_lane_list(data_us101_v_sorted_2, data_us101_frame_2, data_us101_frame_2[0][0][1])
vehicles_change_lane_list_3, change_lane_label_3 = get_change_lane_list(data_us101_v_sorted_3, data_us101_frame_3, data_us101_frame_3[0][0][1])


def get_change_lane_frames(vehicles_change_lane_list, data_us101_frame, offset, pred_n=20, succ_n=21):
    change_lane_frames = [[]]
    for v in vehicles_change_lane_list:
        frame_id = v[2]
        frame = []
        for i in range(pred_n):
            frame.append(data_us101_frame[frame_id-offset-pred_n+i])
        if succ_n != 0:
            for i in range(succ_n):
                frame.append(data_us101_frame[frame_id-offset+i])
        # frame_id -= 10
        change_lane_frames.append((v[0],v[1],frame))
    del change_lane_frames[0]
    # print(change_lane_frames[0])
    return change_lane_frames

change_lane_frames_1 = get_change_lane_frames(vehicles_change_lane_list_1, data_us101_frame_1, data_us101_frame_1[0][0][1])
change_lane_frames_2 = get_change_lane_frames(vehicles_change_lane_list_2, data_us101_frame_2, data_us101_frame_2[0][0][1])
change_lane_frames_3 = get_change_lane_frames(vehicles_change_lane_list_3, data_us101_frame_3, data_us101_frame_3[0][0][1])

print(len(change_lane_frames_1))
print(len(change_lane_frames_1[0][2]))
print(len(change_lane_frames_2))
print(len(change_lane_frames_3))


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

    def show(self, delta=0.05, color='red'):
        self.V1 += delta
        self.V2 = np.add(self.V2, [-delta, delta])
        self.V3 = np.add(self.V3, [-delta, -delta])
        self.V4 = np.add(self.V4, [delta, -delta])
        self.V = np.array([self.V1, self.V2, self.V3, self.V4])
        for n in range(self.N):
            if n < self.N - 1:
                x = [self.V[n][0], self.V[n+1][0]]
                y = [self.V[n][1], self.V[n+1][1]]
            else:
                x = [self.V[n][0], self.V[0][0]]
                y = [self.V[n][1], self.V[0][1]]
            plt.plot(x, y, "r-")
        self.V1 -= delta
        self.V2 = np.add(self.V2, [delta, -delta])
        self.V3 = np.add(self.V3, [delta, delta])
        self.V4 = np.add(self.V4, [-delta, delta])
        self.V = np.array([self.V1, self.V2, self.V3, self.V4])
        for n in range(self.N):
            if n < self.N - 1:
                x = [self.V[n][0], self.V[n + 1][0]]
                y = [self.V[n][1], self.V[n + 1][1]]
            else:
                x = [self.V[n][0], self.V[0][0]]
                y = [self.V[n][1], self.V[0][1]]
            plt.plot(x, y, "g-", linewidth=0.5)


class World:
    def __init__(self, x_min, x_max, v_min, v_max, Pset, vehicle_i, vehicle_g, delta=0.0):
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max
        self.Pset = Pset
        self.vehicle_i = vehicle_i
        self.vehicle_g = vehicle_g
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
        front_edge_center = np.add([s_q[0], s_q[1]], (self.vehicle_i[1] / 2) * norm_vec)
        rear_edge_center = np.add([s_q[0], s_q[1]], (self.vehicle_i[1] / 2) * -norm_vec)
        v1 = np.add(front_edge_center, self.vehicle_i[0] / 2 * norm_vec_orth1)
        v2 = np.add(front_edge_center, self.vehicle_i[0] / 2 * norm_vec_orth2)
        v3 = np.add(rear_edge_center, self.vehicle_i[0] / 2 * norm_vec_orth1)
        v4 = np.add(rear_edge_center, self.vehicle_i[0] / 2 * norm_vec_orth2)
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

    def isVehicle_i(self, s_q):
        if self.vehicle_i.isInside(s_q[0:2]):
            return True
        return False

    def isVehicle_g(self, s_q):
        if self.vehicle_g.isInside(s_q[0:2]):
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

    def show(self):
        p1 = [self.x_min[0], self.x_min[1]]
        p2 = [self.x_min[0], self.x_max[1]]
        p3 = [self.x_max[0], self.x_max[1]]
        p4 = [self.x_max[0], self.x_min[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-")
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], "k-")
        plt.plot([p3[0], p4[0]], [p3[1], p4[1]], "k-")
        plt.plot([p4[0], p1[0]], [p4[1], p1[1]], "k-")
        for P in self.Pset:
            P.show(delta=self.delta)


def find_neighbors(host_id, host_t_f, vehicles, r):
    neighbors = [[]]
    for v in vehicles:
        if v[0] == host_id and v[2] == host_t_f:
            x = [v[4], v[5]]
            lane_id = v[13]
            host = v
            break
    for v in vehicles:
        if v[0] != host_id and v[2] != host_t_f:
            if ((x[1] < v[5] and distance(x, [v[4], v[5]]) < r) or (x[1] > v[5] and distance(x, [v[4], v[5]]) < r-50)) \
                    and (v[13]==lane_id or v[13]==lane_id+1 or v[13]==lane_id-1) and v[13] < 6:
                neighbors.append(v)
    del neighbors[0]
    return neighbors, host


neighbors_1=[[]]
host_1 = []
for v in change_lane_frames_1:
    host_id, host_t_f = v[0], v[1]
    r = 250
    for f in v[2]:
        neighbors_1.append(find_neighbors(host_id, host_t_f, f, r)[0])
        host_1.append(find_neighbors(host_id, host_t_f, f, r)[1])
del neighbors_1[0]
print(len(neighbors_1))
print(len(host_1))


neighbors_2=[[]]
host_2 = []
for v in change_lane_frames_2:
    host_id, host_t_f = v[0], v[1]
    r = 250
    for f in v[2]:
        neighbors_2.append(find_neighbors(host_id, host_t_f, f, r)[0])
        host_2.append(find_neighbors(host_id, host_t_f, f, r)[1])
del neighbors_2[0]

neighbors_3=[[]]
host_3 = []
for v in change_lane_frames_3:
    host_id, host_t_f = v[0], v[1]
    r = 250
    for f in v[2]:
        neighbors_3.append(find_neighbors(host_id, host_t_f, f, r)[0])
        host_3.append(find_neighbors(host_id, host_t_f, f, r)[1])
del neighbors_3[0]

neighbors_1 = np.array(neighbors_1)
neighbors_2 = np.array(neighbors_2)
neighbors_3 = np.array(neighbors_3)

print(neighbors_1.shape)
print(neighbors_2.shape)
print(neighbors_3.shape)


def decompose_vel(x0, x1):
    dx = x1[0] - x0[0]
    dy = x1[1] - x0[1]
    if dx == 0:
        dx = 1e-6
    angle = np.arctan(dy/dx)
    if angle < 0:
        angle = np.pi + angle
    return [x0[0], x0[1], x0[2]*np.cos(angle), x0[2]*np.sin(angle)]

n_host_1 = 489
n_host_2 = 382
n_host_3 = 405
def get_init_goal(host, n):
    host = host.reshape(n, 41)
    init_goal = [[]]
    for i in range(n):
        v = host[i]
        init_x0 = [v[0][4], v[0][5], v[0][11]]
        init_x1 = [v[1][4], v[1][5], v[1][11]]
        goal_x0 = [v[39][4], v[39][5], v[39][11]]
        goal_x1 = [v[40][4], v[40][5], v[40][11]]
        init = decompose_vel(init_x0, init_x1)
        goal = decompose_vel(goal_x0, goal_x1)
        init_goal.append([init, goal])
    del init_goal[0]
    return init_goal

def get_waypoint_with_decomp_vel(host, n):
    host = host.reshape(n, 41)
    new_traj = [[]]
    for i in range(n):
        v = host[i]
        traj = [[]]
        for j in range(40):
            x0 = [v[j][4], v[j][5], v[j][11]]
            x1 = [v[j+1][4], v[j+1][5], v[j+1][11]]
            new_x0 = decompose_vel(x0, x1)
            traj.append(new_x0)
        del traj[0]
        new_traj.append(traj)
    del new_traj[0]
    return new_traj

host_1 = np.array(host_1)
init_goal_1 = get_init_goal(host_1, n_host_1)
i = 0
for v in init_goal_1:
    if v[0][3] <= 0 or v[1][3] <= 0:
        del init_goal_1[i]
        n_host_1 -= 1
        for j in range(41):
            host_1 = np.delete(host_1, i*41)
            neighbors_1 = np.delete(neighbors_1, i*41)
    i += 1
new_host_1 = get_waypoint_with_decomp_vel(host_1, n_host_1)
print(len(new_host_1))

host_2 = np.array(host_2)
init_goal_2 = get_init_goal(host_2, n_host_2)
i = 0
for v in init_goal_2:
    if v[0][3] <= 0 or v[1][3] <= 0:
        del init_goal_2[i]
        n_host_2 -= 1
        for j in range(41):
            host_2 = np.delete(host_2, i*41)
            neighbors_2 = np.delete(neighbors_2, i*41)
    i += 1
new_host_2 = get_waypoint_with_decomp_vel(host_2, n_host_2)
print(len(new_host_2))

host_3 = np.array(host_3)
init_goal_3 = get_init_goal(host_3, n_host_3)
i = 0
for v in init_goal_3:
    if v[0][3] <= 0 or v[1][3] <= 0:
        del init_goal_3[i]
        n_host_3 -= 1
        for j in range(41):
            host_3 = np.delete(host_3, i*41)
            neighbors_3 = np.delete(neighbors_3, i*41)
    i += 1
new_host_3 = get_waypoint_with_decomp_vel(host_3, n_host_3)
print(len(new_host_3))


init_goal_1 = [[]]
for traj in new_host_1:
    init_goal = []
    for i in range(31):
        init_goal.append([traj[i], traj[-1]])
    init_goal_1.append(init_goal)
del init_goal_1[0]
print(len(init_goal_1))
print(len(init_goal_1[0]))

init_goal_2 = [[]]
for traj in new_host_2:
    init_goal = []
    for i in range(31):
        init_goal.append([traj[i], traj[-1]])
    init_goal_2.append(init_goal)
del init_goal_2[0]
print(len(init_goal_2))
print(len(init_goal_2[0]))

init_goal_3 = [[]]
for traj in new_host_3:
    init_goal = []
    for i in range(31):
        init_goal.append([traj[i], traj[-1]])
    init_goal_3.append(init_goal)
del init_goal_3[0]
print(len(init_goal_3))
print(len(init_goal_3[0]))


def generate_occ_grid(host_i, host_g, neighbors, r, left_edge=0, right_edge=60):
    lane1_c = 7.231027419583393
    lane2_c = 18.84918099161813
    lane3_c = 29.990037979406054
    lane4_c = 40.99582664476257
    lane5_c = 53.099915981198585
    lane_c = [7.231027419583393, 18.84918099161813, 29.990037979406054, 40.99582664476257, 53.099915981198585]

    center = [host_i[4], host_i[5]]
    lower = center[1] - 150 - 10
    upper = center[1] + 150 + 10

    if host_i[13] == 1:
        x_0 = 0 - 1
        x_1 = 40 - 1
    if host_i[13] == 2:
        x_0 = 0 - 1
        x_1 = 40 - 1
    if host_i[13] == 3:
        x_0 = 10 - 1
        x_1 = 50 - 1
    if host_i[13] == 4:
        x_0 = 20 - 1
        x_1 = 60 - 1
    if host_i[13] == 5:
        x_0 = 30 - 1
        x_1 = 70 - 1

    x_min = [x_0, lower]
    x_max = [x_1, upper]
    v_min = [-10, -10]
    v_max = [10, 100]

    vehicle_i = Rectangle((host_i[4], host_i[5]), host_i[9], host_i[8])
    vehicle_g = Rectangle((host_g[4], host_g[5]), host_g[9], host_g[8])

    final_Occs = []

    for frame in neighbors:
        Pset = []
        for v in frame:
            Pset.append(Rectangle((v[4], v[5]), v[9], v[8]))
        W = World(x_min, x_max, v_min, v_max, Pset, vehicle_i, vehicle_g, delta=0.0)
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

"""
OccGrids_1 = []
i = 10
j = 10
print("start")
while i < len(neighbors_1):
# while i < 10:
    if i % 100 == 0:
        print(i)
    if i != 0 and i % 41 == 40:
        continue
    predict_v = neighbors_1[i:i + 10]
    Occ, c, v, x_min, x_max = generate_occ_grid(host_1[i], host_1[(int(i/41)+1)*41-1], predict_v, 150)
    for f in range(10):
        np.savetxt("OccGrids_1_10f/OccGrids_1_" + str(j) + "-"+str(f)+".txt", Occ[f])
    j += 1
    # print("x_min", x_min)
    # print("x_max", x_max)
    OccGrids_1.append(Occ)
    if (i + 10) % 41 == 40:
        i += 11
    else:
        i += 1

OccGrids_1 = np.array(OccGrids_1)
print(OccGrids_1[0].shape)
print(len(OccGrids_1))


# OccGrids_2 = []
i = 0
j = 0
print("start")
while i < len(neighbors_2):
# while i < 10:
    if i % 100 == 0:
        print(i)
    if i != 0 and i % 41 == 40:
        continue
    predict_v = neighbors_2[i:i + 10]
    Occ, c, v, x_min, x_max = generate_occ_grid(host_2[i], host_2[(int(i/41)+1)*41-1], predict_v, 150)
    for f in range(10):
        np.savetxt("OccGrids_2_10f/OccGrids_2_" + str(j) + "-"+str(f)+".txt", Occ[f])
    j += 1
    # print("x_min", x_min)
    # print("x_max", x_max)
    # OccGrids_2.append(Occ)
    if (i + 10) % 41 == 40:
        i += 11
    else:
        i += 1

# OccGrids_1 = np.array(OccGrids_2)
# print(OccGrids_1[0].shape)
# print(len(OccGrids_1))
"""


i = 0
j = 0
print("start")
while i < len(neighbors_3):
# while i < 10:
    if i % 100 == 0:
        print(i)
    if i != 0 and i % 41 == 40:
        continue
    predict_v = neighbors_3[i:i + 10]
    Occ, c, v, x_min, x_max = generate_occ_grid(host_3[i], host_3[(int(i/41)+1)*41-1], predict_v, 150)
    for f in range(10):
        np.savetxt("OccGrids_3_10f/OccGrids_3_" + str(j) + "-"+str(f)+".txt", Occ[f])
    j += 1
    # print("x_min", x_min)
    # print("x_max", x_max)
    # OccGrids_2.append(Occ)
    if (i + 10) % 41 == 40:
        i += 11
    else:
        i += 1

