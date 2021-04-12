import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Bidirectional
from keras.models import load_model
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from utilize import *
import time 
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# Device
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
KTF.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# save trajectory images
save_less=30 # num of images 1
save_more=100  # num of images 2
save_train_cases=False 
save_result_cases=False
save_m=True # save model

cate="" # which data set to use. Becaus I utilize different data sets to train stage-1 model, stage-2 model and Social LSTM model.
#Simply set it to "" if use the general data set to train LSTM model

# learning params
continue_flag=False
model_path="./checkpoints/best_weights"+cate+".h5"
EPOCH=60
BATCH_SIZE=128
LR=0.000125
# LR=0.000125/2
# LR=0.000125/4

# settings
train_frames=10
# attr_list=[2,3,4,5,6,8] #namely, Local_X,Local_Y,acceleration,velocity,Lane_ID,heading angle theta
attr_list=[2,3,5,6,7,9]
attr_num=len(attr_list)
class_num=3
TRAIN_NUM=10000

# save directory
save_img_dir="./img_"+cate+"/"
if not os.path.exists(save_img_dir):
    os.mkdir(save_img_dir)
save_model_dir="./checkpoints/"
model_name="best_weights"+cate+".h5"
# model_name="best_weights_re"+cate+".h5"

dataset_root="./data/"
specific1="train"
specific2="train"
df = pd.read_csv(dataset_root+specific1+"/train"+cate+".csv")
df2=pd.read_csv(dataset_root+specific2+"/train"+cate+".csv")
print(df.shape)
print(df2.shape)

#label position = 9 
label=df.iloc[:,-1].values
label2=df2.iloc[:,-1].values

data_num=int(len(label)/train_frames)
data_num2=int(len(label2)/train_frames)
data=df.iloc[:,attr_list].values
data2=df2.iloc[:,attr_list].values
data=np.reshape(data,[data_num,train_frames,attr_num])
data2=np.reshape(data2,[data_num2,train_frames,attr_num])
print("Data_number for training:",data_num)
print("Data_number for testing:",data_num2)
#add label
train_label=pro_list(data_num=data_num,train_list=label,pred=train_frames)
test_label=pro_list(data_num=data_num2,train_list=label2,pred=train_frames)

#print proportion
#0 = keep lane, 1 = turn left, 2 = turn right
#train arrays
check_train_data_l=[]
check_train_data_r=[]
check_train_data_kl=[]

turn_left_train=[]
turn_right_train=[]
keep_lane_train=[]
cnt0=cnt1=cnt2=0
for i in range(len(train_label)):
    if train_label[i]==0:
        cnt0+=1
        keep_lane_train.append(data[i])
        check_train_data_kl.append(i)
    elif train_label[i]==1:
        cnt1+=1
        turn_left_train.append(data[i])
        check_train_data_l.append(i)
    elif train_label[i]==2:
        cnt2+=1
        turn_right_train.append(data[i])
        check_train_data_r.append(i)
print("Train-record Proportion:(keep lane, left, right)",cnt0,cnt1,cnt2)



#save training case for inspection
#=============================save training case for inspection============================= can just ignore
if(save_train_cases==True):
    print("Saving training data(left):")
    if not os.path.exists(save_img_dir+"train_l"):
        os.mkdir(save_img_dir+"train_l")
    for i in range(save_less):
        q_id=check_train_data_l[i]
        case1=df[q_id*train_frames:(q_id+1)*train_frames]
        fig = plt.figure()
        for j in range(train_frames):
            plt.scatter(case1['Local_X'].values[j],case1['Local_Y'].values[j])
        fig.suptitle(str(case1['Vehicle_ID'].values[-1])+' '+str(case1['Frame_ID'].values[-1]))
        plt.savefig(save_img_dir+"train_l/l{}.png".format(i))
        plt.close(fig)
    print("Saving training data(right):")
    if not os.path.exists(save_img_dir+"train_r"):
        os.mkdir(save_img_dir+"train_r")
    for i in range(save_less):
        q_id=check_train_data_r[i]
        case1=df[q_id*train_frames:(q_id+1)*train_frames]
        fig = plt.figure()
        for j in range(train_frames):
            plt.scatter(case1['Local_X'].values[j],case1['Local_Y'].values[j])
        fig.suptitle(str(case1['Vehicle_ID'].values[-1])+' '+str(case1['Frame_ID'].values[-1]))
        plt.savefig(save_img_dir+"train_r/r{}.png".format(i))
        plt.close(fig)
    print("Saving training data(keep lane):")
    if not os.path.exists(save_img_dir+"train_kl"):
        os.mkdir(save_img_dir+"train_kl")
    for i in range(save_less):
        q_id=check_train_data_kl[i]
        case1=df[q_id*train_frames:(q_id+1)*train_frames]
        fig = plt.figure()
        for j in range(train_frames):
            plt.scatter(case1['Local_X'].values[j],case1['Local_Y'].values[j])
        fig.suptitle(str(case1['Vehicle_ID'].values[-1])+' '+str(case1['Frame_ID'].values[-1]))
        plt.savefig(save_img_dir+"train_kl/kl{}.png".format(i))
        plt.close(fig)


#test arrays
test_order0=[]
test_order1=[]
test_order2=[]

turn_left_test=[]
turn_right_test=[]
keep_lane_test=[]
cnt0=cnt1=cnt2=0
for i in range(len(test_label)):
    if test_label[i]==0:
        cnt0+=1
        keep_lane_test.append(data2[i])
        test_order0.append(i)
    elif test_label[i]==1:
        cnt1+=1
        turn_left_test.append(data2[i])
        test_order1.append(i)
    elif test_label[i]==2:
        cnt2+=1
        turn_right_test.append(data2[i])
        test_order2.append(i)
print("Test-record Proportion:(keep lane, left, right)",cnt0,cnt1,cnt2)

#data balanced
#num=min(len(keep_lane_train),len(turn_left_train),len(turn_right_train))
num=len(keep_lane_train)
#fetch data frome the pool
print("Training data description:")
d0_train,l0_train=sampling_from_pool(keep_lane_train,[[1,0,0]],num)
d1_train,l1_train=sampling_from_pool(turn_left_train,[[0,1,0]],num)
d2_train,l2_train=sampling_from_pool(turn_right_train,[[0,0,1]],num)
data,train_label=combineData(d0_train, l0_train, d1_train, l1_train, d2_train, l2_train,"Train")

print("Testing data description:")
d0_test,l0_test=sampling_from_pool(keep_lane_test,[[1,0,0]],len(keep_lane_test))
d1_test,l1_test=sampling_from_pool(turn_left_test,[[0,1,0]],len(turn_left_test))
d2_test,l2_test=sampling_from_pool(turn_right_test,[[0,0,1]],len(turn_right_test))
test_order=test_order0+test_order1+test_order2
print(test_order[-20:])
data2,test_label=combineData(d0_test,l0_test,d1_test,l1_test,d2_test, l2_test,"Test")

def get_data(data,label,random_sample=False):
    data,label=shuffle(data,label)
    return data,label
def get_train_data(data,label,train_data_num=TRAIN_NUM,random_sample=False):
    data,label=shuffle(data,label)
    return data,label
def get_test_data(data,label,random_sample=False):
    return data,label


def buildManyToOneModel(shape,class_num,LR):
  model = Sequential()
  # model.add(LSTM(128, input_length=shape[1], input_dim=shape[2]))
  # model.add(Dropout(0.5))
  model.add(LSTM(128, input_shape=(shape[1],shape[2]),return_sequences=True))
  model.add(Dropout(0.5))
  model.add(LSTM(128,return_sequences=False))
  model.add(Dropout(0.5))
  model.add(Dense(class_num,activation='softmax'))

  op=keras.optimizers.Adam(lr=LR)
  model.compile(loss="categorical_crossentropy", optimizer=op, metrics=['accuracy'])
  model.summary()
  return model

def buildManyToOneModel_single(shape,class_num,LR):
  model = Sequential()
  # model.add(LSTM(128, input_length=shape[1], input_dim=shape[2]))
  # model.add(Dropout(0.5))
  model.add(LSTM(128, input_shape=(shape[1],shape[2]),return_sequences=False))
  model.add(Dropout(0.5))
  model.add(Dense(class_num,activation='softmax'))

  op=keras.optimizers.Adam(lr=LR)
  model.compile(loss="categorical_crossentropy", optimizer=op, metrics=['accuracy'])
  model.summary()
  return model

def buildManyToOneModel_Bi(shape,class_num,LR):
  print("Bidirectional model")
  model = Sequential()
  model.add(Bidirectional(LSTM(128,return_sequences=True),input_shape=(shape[1],shape[2])))
  model.add(Dropout(0.5))
  model.add(LSTM(128,return_sequences=False))
  model.add(Dropout(0.5))
  model.add(Dense(class_num,activation='softmax'))

  op=keras.optimizers.Adam(lr=LR)
  model.compile(loss="categorical_crossentropy", optimizer=op, metrics=['accuracy'])
  model.summary()
  return model

#self split into 2 set or use trainset and testset
# if(specific2==specific1):
#     X,y=get_data(data=data,label=train_label)
#     x_train,x_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print("Train and test composition:")
#     print(len(X))
#     print(len(x_train))
#     print(len(x_test))
# else:
x_train,y_train=get_train_data(data=data,label=train_label,train_data_num=TRAIN_NUM)
x_test,y_test=get_test_data(data=data2,label=test_label)
print("Train and test composition:")
print(len(x_train))
print(len(x_test))
print(y_train[:10])
if(continue_flag==True):
    model=load_model(model_path)  
else:
    # model = buildManyToOneModel(x_train.shape,class_num,LR)
    model = buildManyToOneModel_Bi(x_train.shape,class_num,LR)

#training process
t1=time.time()
def scheduler(epoch):
    if epoch % 35 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)
if(save_m==True):
    checkpoint = ModelCheckpoint(filepath=save_model_dir+model_name,monitor='val_acc',mode='auto' ,save_best_only='True')
    callback_lists=[checkpoint]
    model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE,validation_data=(x_test, y_test),callbacks=[checkpoint])
else:
    model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE,validation_data=(x_test, y_test),callbacks=[])
t2=time.time()
print(t2-t1)
result=model.predict(x_test)

#testing process
loss, accuracy = model.evaluate(x_test,y_test)
print("Loss, Acc:",loss,accuracy)
true_category=np.argmax(y_test,axis=1)
pred_category=np.argmax(result,axis=1)
m=confusion_matrix(true_category,pred_category)
print(m)
acc_list=[]
for i in range(len(m)):
    tot=0
    for j in range(len(m[i])):
        tot+=m[i][j]
    acc=float(m[i][i])/float(tot)
    acc_list.append(acc)
print(acc_list)

print("Count of absolute possibility value:")
#count for absolute value
fg1=0
fg2=0
for i in range(len(result)):
    if(np.max(result[i])>=0.8):
        fg1+=1
    else:
        fg2+=1
print(fg1,fg2)