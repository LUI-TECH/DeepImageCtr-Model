import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, BatchNormalization, Input,Concatenate,Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from sklearn.datasets import load_files
from glob import glob
import matplotlib
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Zeros, glorot_normal
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import sklearn
from sklearn.preprocessing import LabelEncoder
from input_deepctr import *
from AttentionPoolingLayer import *

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def img_prepare(img_dict,Urls):
    img_tensor_dict={}
    for i in img_dict:
        img_tensor_dict[img_dict[i]]=preprocess_input(path_to_tensor(img_dict[i]))
    img_tensors_ad=[img_tensor_dict[i] for i in Urls]
    img_input_ad=np.array([img_tensors_ad[i][0] for i in range(len(img_tensors_ad))])
    return img_input_ad



#encode string inputs to integer 
def Encoder(lables):
    enc = LabelEncoder()
    enc.fit(lables)
    encoded=enc.transform(lables)
    return np.array(encoded)

# Test when there is no hist inputs
#def Encoder_list(lables):
#    enc = LabelEncoder()
#    enc.fit(lables)
#    encoded=enc.transform(lables)
#    return np.array(encoded)
#def histEncoder(lables1,lables2,lables3,lables4):
#    lables1=Encoder_list(lables1)
#    lables2=Encoder_list(lables2)
#    lables3=Encoder_list(lables3)
#    lables4=Encoder_list(lables4)
#    Lable=list()
#    for i in range(len(lables1)):
#      Lable.append([lables1[i],lables2[i],lables3[i],lables4[i]])
#    return np.array(Lable)

def histEncoder(lables):
    all_element=[]
    Encoded=[]
    for i in lables:
        all_element+=i
    enc = LabelEncoder()
    enc.fit(all_element)
    for i in lables:
        encoded=enc.transform(i)
        Encoded.append(encoded)
    return np.array(Encoded)



def get_xy_fd(target,uid,gender,age,province,grade,interest,item_id,item_category,ad_pos,pool_id,hist_item_id,hist_item_category,img_list,img_dict):
    interest1,interest2,interest3,interest4,interest5,interest6,interest7,interest8,interest9,interest10=interest_input(interest)
    category1,category2,category3 = category_input(item_category)
    ad_img=img_prepare(img_dict,img_list)
    
    # create objects for each inputs
    feature_columns = [SparseFeat('uid',len(set(uid))),SparseFeat('user_gender', len(set(gender))), 
                       SparseFeat('user_age',len(set(age))),#SparseFeat('target', len(set(target))),
                       SparseFeat('user_province',len(set(province))),SparseFeat('user_interest1', len(set(interest))),
                       SparseFeat('user_interest2', len(set(interest))),SparseFeat('user_interest3', len(set(interest))),
                       SparseFeat('user_interest4', len(set(interest))),SparseFeat('user_interest5', len(set(interest))),
                       SparseFeat('user_interest6', len(set(interest))),SparseFeat('user_interest7', len(set(interest))),
                       SparseFeat('user_interest8', len(set(interest))),SparseFeat('user_interest9', len(set(interest))),
                       SparseFeat('user_interest10', len(set(interest))),
                       SparseFeat('item_id', len(set(item_id))), SparseFeat('item_category1', len(set(item_category))),
                       SparseFeat('item_category2', len(set(item_category))),SparseFeat('item_category3', len(set(item_category))),
                       SparseFeat('adpos', len(set(ad_pos))), SparseFeat('pool_id', len(set(pool_id))),
                       ImgDenseFeat('item_graph')]
    feature_columns += [VarLenSparseFeat('hist_item_id',len(set(hist_item_id)), maxlen=4, embedding_name='item_id'),
                        VarLenSparseFeat('hist_item_category1',len(set(hist_item_category1)), maxlen=4, embedding_name='item_category1'),
                        VarLenSparseFeat('hist_item_category2',len(set(hist_item_category2)), maxlen=4, embedding_name='item_category2'),
                        VarLenSparseFeat('hist_item_category3',len(set(hist_item_category3)), maxlen=4, embedding_name='item_category3')]
    
    # define the names of inputs that will be used for attention pooling 
    behavior_feature_list = ["item_id", "item_category1","item_category2","item_category3"]

    # encoding string inputs
    target=Encoder(target)
    uid=Encoder(uid)
    gender=Encoder(gender)
    age=Encoder(age)
    province=Encoder(province)
    grade=Encoder(grade)
    interest1=Encoder(interest1)
    interest2=Encoder(interest2)
    interest3=Encoder(interest3)
    interest4=Encoder(interest4)
    interest5=Encoder(interest5)
    interest6=Encoder(interest6)
    interest7=Encoder(interest7)
    interest8=Encoder(interest8)
    interest9=Encoder(interest9)
    interest10=Encoder(interest10)
    item_id=Encoder(item_id)
    category1=Encoder(category1)
    category2=Encoder(category2)
    category3=Encoder(category3)
    adpos=Encoder(ad_pos)
    pool_id=Encoder(pool_id)
    # encode historical inputs
    hist_iid = histEncoder(hist_item_id)
    hist_category1 = histEncoder(hist_item_category1)
    hist_category2 = histEncoder(hist_item_category2)
    hist_category3 = histEncoder(hist_item_category3)


    # Create input dictionary
    feature_dict = {'uid': uid, 'user_gender': gender,'user_age': age,#'target': target,
                    'user_province':province, 'user_interest1':interest1,'user_interest2':interest2,
                    'user_interest3':interest3,'user_interest4':interest4,'user_interest5':interest5,
                    'user_interest6':interest6,'user_interest7':interest7,'user_interest8':interest8,
                    'user_interest9':interest9,'user_interest10':interest10,
                    'item_id': item_id, 
                    'item_category': category1, 'item_category2': category2,'item_category3': category3,
                    'adpos':adpos,'pool_id':pool_id,'hist_item_id': hist_iid,
                    'hist_item_category1': hist_category1,'hist_item_category2': hist_category2,
                    'hist_item_category3': hist_category3,'item_graph': ad_img}
    fixlen_feature_names = get_fixlen_feature_names(feature_columns)
    varlen_feature_names = get_varlen_feature_names(feature_columns)


    # Input and groundtruth
    # Currently use target as the final prediction (IMP or Clk)
    x = [feature_dict[name] for name in fixlen_feature_names] + [feature_dict[name] for name in varlen_feature_names]
    x_test = [feature_dict[name][:30] for name in fixlen_feature_names] + [feature_dict[name][:30] for name in varlen_feature_names]
    y_test=target[:30]
    y = target
    return x, y, feature_columns, behavior_feature_list,x_test,y_test





def category_input(category):
    #divide categorical input into 3 separate inputs 
    category1=[]
    category2=[]
    category3=[]
    for i in category:
        j=i.strip().split('|',-1)
        if len(j)==3:
          category1.append(j[0])
          category2.append(j[1])
          category3.append(j[2])
        else:
            try:
                category1.append(j[0])
            except:
                category1.append('0')
            try:
                category2.append(j[1])
            except:
                category2.append('0')
            try:
                category3.append(j[2])
            except:
                category3.append('0')
    return category1,category2,category3
def interest_input(interest):
    #divide interest input into 10 separate inputs
    interest1=[]
    interest2=[]
    interest3=[]
    interest4=[]
    interest5=[]
    interest6=[]
    interest7=[]
    interest8=[]
    interest9=[]
    interest10=[]
    for i in interest:
        j=i.strip().split('|',-1)
        if len(j)==10:
          interest1.append(j[0])
          interest2.append(j[1])
          interest3.append(j[2])
          interest4.append(j[3])
          interest5.append(j[4])
          interest6.append(j[5])
          interest7.append(j[6])
          interest8.append(j[7])
          interest9.append(j[8])
          interest10.append(j[9])
        else:
            try:
                interest1.append(j[0])
            except:
                interest1.append('0')
            try:
                interest2.append(j[1])
            except:
                interest2.append('0')
            try:
                interest3.append(j[2])
            except:
                interest3.append('0')
            try:
                interest4.append(j[3])
            except:
                interest4.append('0')               
            try:
                interest5.append(j[4])
            except:
                interest5.append('0')
            try:
                interest6.append(j[5])
            except:
                interest6.append('0')
            try:
                interest7.append(j[6])
            except:
                interest7.append('0')               
            try:
                interest8.append(j[7])
            except:
                interest8.append('0')
            try:
                interest9.append(j[8])
            except:
                interest9.append('0')
            try:
                interest10.append(j[9])
            except:
                interest10.append('0')               
    return interest1,interest2,interest3,interest4,interest5,interest6,interest7,interest8,interest9,interest10






