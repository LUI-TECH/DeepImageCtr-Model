# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, BatchNormalization, Input,Concatenate,Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
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
import os
import copy
from input_deepctr import *
from AttentionPoolingLayer import *
from preProcession import *
from Model_Input_Procession import *
import argparse
import sys
#---------------------------------------------------MODEL--------------------------------------------------------------
def DeepCtr_Model(dnn_feature_columns,history_feature_list, embedding_size=8, hist_len_max=16, dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024,
        task='binary'):
    """Instantiates the Deep Interest Network architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    # prepare basic user&item info
    features = build_input_features(dnn_feature_columns)
    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(filter(lambda x:isinstance(x, DenseFeat),dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x:isinstance(x,VarLenSparseFeat),dnn_feature_columns)) if dnn_feature_columns else []
    # add historical data of the user interms of item info visited 
    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    # create list of Input tensor object for NN
    inputs_list = list(features.values())
    # nput embedding ID local attention pooling
    embedding_dict = create_embedding_matrix(dnn_feature_columns,l2_reg_embedding,init_std,seed,embedding_size, prefix="")
    
    # match embedding with Input tensor object interms of key and query for attentive pooling
    query_emb_list = embedding_lookup(embedding_dict,features,sparse_feature_columns,history_feature_list,history_feature_list)#query是单独的
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names, history_fc_names)
    # concate to final input for NN
    keys_emb = Concatenate(axis=-1)(keys_emb_list)
    query_emb = Concatenate(axis=-1)(query_emb_list)
    # attention pooling of ID info and hist ID info
    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
        weight_normalization=att_weight_normalization,
        supports_masking=False)([query_emb, keys_emb])
    hist=Flatten_Dense(hist,128)
    print("ID part succeed","the resultant tensor is :",hist)
    # img features
    input_ad=features['item_graph']
    img_tensor_ad=bottle_neck(input_ad)
    img_tensor_ad = tf.keras.layers.Lambda(lambda x: tf.reshape(x,[tf.shape(img_tensor_ad)[0],1,img_tensor_ad.shape[1]]),output_shape=(tf.shape(img_tensor_ad)[0],1,img_tensor_ad.shape[1]))(img_tensor_ad)
    print("Image part succeed","the resultant tensor is :",img_tensor_ad)
    #attention_result = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
    #                                     weight_normalization=att_weight_normalization, supports_masking=False)([
    #    hist, hist_img])   
    #print("Combined attention succeed","the resultant tensor is :",attention_result)
    
    # Feed to DNN
    dnn_input_emb_list = embedding_lookup(embedding_dict,features,sparse_feature_columns,
                                          mask_feat_list=history_feature_list)
    deep_input_emb = Concatenate(axis=-1)(dnn_input_emb_list)
  
    deep_input_emb = Concatenate()([NoMask()(deep_input_emb), hist,img_tensor_ad])
    print("Concation result:",deep_input_emb)
    
    deep_input_emb = Flatten()(deep_input_emb)
    print("Flatten result:",deep_input_emb)
    deep_input_emb=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                   beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                   beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(deep_input_emb)
    print("BatchNorm result:",deep_input_emb)
    #dnn_input = combined_dnn_input([deep_input_emb],dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,dnn_dropout, dnn_use_bn, seed)(deep_input_emb)#(dnn_input)
    print("DNN result:",output)
    final_logit = Dense(1, use_bias=False)(output)
    print("only one output",final_logit)
    output = PredictionLayer(task)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model
def bottle_neck(ipt):
    # add several layers to the pre-trained ResNet50
    base_model = ResNet50(input_tensor=ipt,weights='/cephfs/group/teg-openrecom-openrc/louisbjmao/ft_local/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)
    for layer in base_model.layers[:-11]:
        layer.trainable = False
    x = base_model.output
    x= GlobalAveragePooling2D()(x)
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                   beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                   beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
    x=Dense(256,activation="relu")(x)
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                   beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                   beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
    x=Dense(128,activation="relu")(x)
    return x

    # simply to feed output from pooling layer to a dense layger
def Flatten_Dense(ts,size):
    fts = Flatten()(ts)
    ts = Dense(size)(fts)
    ts = tf.keras.layers.Lambda(lambda x:tf.reshape(x,[tf.shape(ts)[0],1,ts.shape[1]]),output_shape=(tf.shape(ts)[0],1,ts.shape[1]))(ts)
    return ts





#---------------------------------------------------TRAIN--------------------------------------------------------------
def Derive_Input(root_id,root_img):
	store=read_ID(root_id)
	img_dict=Image_Procession(root_img)
	target,uid,gender,age,province,grade,interest,item_id,item_category,ad_pos,pool_id,img_list=Inputs_preProcess(img_dict,store)
	x, y, feature_columns, behavior_feature_list,x_test,y_test=get_xy_fd(target,uid,gender,age,province,grade,interest,item_id,item_category,ad_pos,pool_id,img_list,img_dict)
	return x, y, feature_columns, behavior_feature_list,x_test,y_test

def main(_):
    Train(FLAGS)

def Train(FLAGS):
    IDpath=FLAGS.IDpath
    IMGpath=FLAGS.IMGpath
    Drop_out=FLAGS.drop_out
    DNN_hidden=FLAGS.n_hidden
    Att_hidden=FLAGS.n_att_hidden
    epochs=FLAGS.max_epochs
    L2_DNN=FLAGS.betaDNN
    L2_Emb=FLAGS.betaATT
    embedding_size=FLAGS.embedding_size
    DNN_activation=FLAGS.DNN_activation
    loss=FLAGS.loss
    BatchNormalization=FLAGS.BatchNormalization
    stddev=FLAGS.stddev
    split=FLAGS.split
    learning_rate=FLAGS.learning_rate
    batch_size=FLAGS.batch_size
    x, y, feature_columns, behavior_feature_list,x_test,y_test=Derive_Input(IDpath,IMGpath)
    model=DeepCtr_Model(feature_columns,behavior_feature_list,embedding_size=embedding_size, hist_len_max=4, dnn_use_bn=BatchNormalization,
        dnn_hidden_units=DNN_hidden, dnn_activation=DNN_activation, att_hidden_size=Att_hidden, att_activation="dice",
        att_weight_normalization=BatchNormalization, l2_reg_dnn=L2_DNN, l2_reg_embedding=L2_Emb, dnn_dropout=Drop_out, init_std=stddev, seed=1024,
        task='binary')
    adam=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss, optimizer=adam,metrics=[loss])
    history = model.fit(x, y,batch_size, verbose=1, epochs=epochs, validation_split=split)
    print(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--IDpath",
        type=str,
        default="",
        help="business_path."
    )
    parser.add_argument(
        "--IMGpath",
        type=str,
        default="",
        help="business_path."
    )
    parser.add_argument(
        "--drop_out",
        type=float,
        default=0,
        help="drop out rate."
    )
    parser.add_argument(
        "--n_hidden",
        type=tuple,
        default=(200,80),
        help="number of hidden layers in DNN"
    )
    parser.add_argument(
        "--n_att_hidden",
        type=tuple,
        default=(80,40),
        help="number of hidden layers in attention pooling layer"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="max epoch"
    )

    parser.add_argument(
        "--betaDNN",
        type=float,
        default=1e-6,
        help="L2 regulariz for DNN"
    )
    parser.add_argument(
        "--betaATT",
        type=float,
        default=1e-6,
        help="L2 regularize for Attention Pooling"
    )

    parser.add_argument(
        "--embedding_size",
        type=int,
        default=8,
        help="embedding_size"
    )

    parser.add_argument(
        "--DNN_activation",
        type=str,
        default='relu',
        help="activation functions"
    )

    parser.add_argument(
        "--loss",
        type=str,
        default='binary_crossentropy',
        help="loss function"
    )

    parser.add_argument(
        "--BatchNormalization",
        type=bool,
        default=True,
        help="loss function"
    )

    parser.add_argument(
        "--stddev",
        type=float,
        default=0.0001,
        help="stddev"
    )

    parser.add_argument(
        "--split",
        type=float,
        default=0.5,
        help="split"
    )


    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="--learning_rate."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="--batch_size."
    )


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
