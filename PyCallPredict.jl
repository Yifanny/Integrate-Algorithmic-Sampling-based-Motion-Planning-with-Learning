using LinearAlgebra
using StaticArrays
const Vec2f = SVector{2, Float64}
const Vec4f = SVector{4, Float64}
const VecFrame = SVector{18, Float64}

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
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.models import load_model
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import time

def get_frames(db,Vehicle_IDs,Frame_ID):#Vehicle_IDs in order, for better performance
    pred_n=10
    cnt=0
    vid=db['Vehicle_ID'].values
    f=db['Frame_ID'].values

    lx=db['Local_X'].values
    ly=db['Local_Y'].values
    v=db['v_Vel'].values
    a=db['v_Acc'].values
    l=db['Lane_ID'].values
    t=db['theta'].values
    frames_for_all_vehicles=[]
    case_list=[]
    distance=[] #if second model, we record the dx between line and current point to distinguish online case

    #Requirement: must know its previous 20 frames
    for i in range(len(db)):
        # print(vid[i])
        if(vid[i]==Vehicle_IDs[cnt] and Frame_ID==f[i]):
            record=[]

            #which model to predict, check previous 20 frames(not includes itself)
            flag=False
            for j in range(20):
                if(l[i-j]!=l[i-j-1]):
                  flag=True
                  distance.append(lx[i]-lx[i-j-1])
                  break
            if(flag==False):#B->before lane change: first model, no need to record distance dx
                case_list.append(0)
                distance.append(None)
            if(flag==True):#A->after lane change: second model
                case_list.append(1)

            #necessary information
            for j in range(pred_n):
                qid=i-pred_n+1+j
                record.append([lx[qid],ly[qid],v[qid],a[qid],l[qid],t[qid]])

            frames_for_all_vehicles.append(record)
            cnt+=1
            if(cnt==len(Vehicle_IDs)):
                break
    # print(len(frames_for_all_vehicles))
    return np.array(frames_for_all_vehicles),np.array(case_list),np.array(distance)

def predict_action(Vehicle_IDs,Frame_ID,single_model=False,noise_threshold=0.5,pred_threshold=0.8):
    pred_n=10

    #test case
    time_stamp="0820am-0835am"
    db=pd.read_csv("pre_data/"+time_stamp+"/add_label.csv")
    # print(len(db))
    single_model=False
    model_dir='./model/'
    model=None
    model0=None#0 means first model before lane change
    model1=None#1 means second model after lane change
    if(single_model):
        model_cate="_np"
        model_name="best_weights"+model_cate+".h5"
        model=load_model(model_dir+model_name)
    else:
        model_cate0="_B"
        model_cate1="_An"
        model_name0="best_weights_npnew"+model_cate0+".h5"
        model_name1="best_weights_npnew"+model_cate1+".h5"
        model0=load_model(model_dir+model_name0)
        model1=load_model(model_dir+model_name1)
    print("Model loaded")
    st1=time.time()
    # print(Vehicle_IDs)
    frames_list,case_list,distance=get_frames(db,Vehicle_IDs,Frame_ID)
    st2=time.time()
    if(single_model==False):
        # print(frames_list.shape)
        result0=model0.predict(frames_list)
        result1=model1.predict(frames_list)
        pred_category0=np.argmax(result0,axis=1)
        pred_category1=np.argmax(result1,axis=1)
        pred_category=(case_list==0).astype(int)*pred_category0+(case_list==1).astype(int)*pred_category1


        second=np.array([-1]*len(pred_category))#find second possible direction, if possibility of first possible direction >0.8, second = -1
        sorted_result0=np.sort(result0,axis=1)
        sorted_result1=np.sort(result1,axis=1)
        for i in range(len(Vehicle_IDs)):
            # print(sorted_result1)
            if(case_list[i]==0 and sorted_result0[i][-1]<pred_threshold):
                for j in range(3):
                    if(result0[i][j]==sorted_result0[i][-2]):
                        second[i]=j
                        break
            if(case_list[i]==1 and sorted_result1[i][-1]<pred_threshold):
                for j in range(3):
                    if(result1[i][j]==sorted_result1[i][-2]):
                        second[i]=j
                        break

    else:
        result=model.predict(frames_list)
        pred_category=np.argmax(result,axis=1)

    decision=[]
    for i in range(len(pred_category)):
        if(case_list[i]==1):#After lane change model

            #right lane change occurs in previous 20 frames
            if(distance[i]>0):
                if(pred_category[i]==2):
                    decision.append(pred_category[i])#turn right case
                elif(distance<=noise_threshold):
                    decision.append(0)#on line case
                else:
                    decision.append(pred_category[i])#keep line case or turn left case


            #left lane change occurs in previous 20 frames
            if(distance[i]<=0):
                if(pred_category[i]==1):
                    decision.append(pred_category[i])#turn left case
                elif(distance>=-noise_threshold):
                    decision.append(0)#on line case
                else:
                    decision.append(pred_category[i])#keep line case or turn right case

        else:#Before lane change model
            decision.append(pred_category[i])
    et=time.time()
    # print("et-st1:",et-st1)
    # print("et-st2:",et-st2)
    # print(decision,second,pred_category)
    # print(case_list,distance,result1,result0)
    return decision,second
"""

function get_Vehicle_IDs(frame::Vector{VecFrame})
    res = Int64[]
    for v in frame
        push!(res, v[1])
    end
    return res
end

function predict_action(prev_frame::Vector{VecFrame}, frame::Vector{VecFrame}, Frame_ID::Int64)
    Vehicle_IDs = get_Vehicle_IDs(frame)
    # println(Vehicle_IDs)
    decisions, seconds = py"predict_action"(Vehicle_IDs, Frame_ID)
    threshold_keep = [-0.075, 0.075]
    threshold_left = [-0.2, 0.0]
    threshold_right = [0.0, 0.2]
    keep_cal = 0.0
    left_cal = -0.1
    right_cal = 0.075
    frame1 = copy(frame)
    frame2 = copy(frame)
    frame3 = copy(frame)
    frame4 = copy(frame)
    frame5 = copy(frame)
    frame6 = copy(frame)
    frame7 = copy(frame)
    frame8 = copy(frame)
    frame9 = copy(frame)
    for i = 1:length(decisions)
        if seconds[i] != -1
            println("lalala")
        end
        v_id = Vehicle_IDs[i]
        decision = decisions[i]
        theta = 0.0
        for v in prev_frame
            if v[1] == v_id
                theta = atan((frame[i][5] - v[5])/(frame[i][6] - v[6]))
                break
            end
        end
        if decision == 0
            if !(threshold_keep[1] <= theta <= threshold_keep[2])
                theta = (theta + keep_cal)/2
            end
        elseif decision == 1 # left
            if !(threshold_left[1] <= theta <= threshold_left[2])
                theta = (theta + left_cal)/2
            end
        elseif decision == 2 # right
            if !(threshold_right[1] <= theta <= threshold_right[2])
                theta = (theta + right_cal)/2
            end
        end
        vx = frame[i][12] * sin(theta)
        vy = frame[i][12] * cos(theta)
        frame1[i] = setindex(frame1[i], frame[i][5] + 0.1 * vx, 5)
        frame2[i] = setindex(frame2[i], frame[i][5] + 0.2 * vx, 5)
        frame3[i] = setindex(frame3[i], frame[i][5] + 0.3 * vx, 5)
        frame4[i] = setindex(frame4[i], frame[i][5] + 0.4 * vx, 5)
        frame5[i] = setindex(frame5[i], frame[i][5] + 0.5 * vx, 5)
        frame6[i] = setindex(frame6[i], frame[i][5] + 0.6 * vx, 5)
        frame7[i] = setindex(frame7[i], frame[i][5] + 0.7 * vx, 5)
        frame8[i] = setindex(frame8[i], frame[i][5] + 0.8 * vx, 5)
        frame9[i] = setindex(frame9[i], frame[i][5] + 0.9 * vx, 5)
        frame1[i] = setindex(frame1[i], frame[i][6] + 0.1 * vy, 6)
        frame2[i] = setindex(frame2[i], frame[i][6] + 0.2 * vy, 6)
        frame3[i] = setindex(frame3[i], frame[i][6] + 0.3 * vy, 6)
        frame4[i] = setindex(frame4[i], frame[i][6] + 0.4 * vy, 6)
        frame5[i] = setindex(frame5[i], frame[i][6] + 0.5 * vy, 6)
        frame6[i] = setindex(frame6[i], frame[i][6] + 0.6 * vy, 6)
        frame7[i] = setindex(frame7[i], frame[i][6] + 0.7 * vy, 6)
        frame8[i] = setindex(frame8[i], frame[i][6] + 0.8 * vy, 6)
        frame9[i] = setindex(frame9[i], frame[i][6] + 0.9 * vy, 6)
    end
    return frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9
end

# Vehicle_IDs=[1,2,3]
# Frame_ID=300
# predict_action(Vehicle_IDs,Frame_ID)
