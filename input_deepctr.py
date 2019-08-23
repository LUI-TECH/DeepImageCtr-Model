import numpy as np
from collections import OrderedDict, namedtuple
from itertools import chain
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import  Embedding, Input, Flatten,Concatenate
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf 




class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype','embedding_name','embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None,embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name,embedding)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):

        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)
      
class ImgDenseFeat(namedtuple('ImgDenseFeat', ['name' , 'dtype'])):
    __slots__ = ()

    def __new__(cls, name,dtype="float32"):
        return super(ImgDenseFeat, cls).__new__(cls, name,dtype)



class VarLenSparseFeat(namedtuple('VarLenFeat', ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype','embedding_name','embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype, embedding_name,embedding)



class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            x = tf.as_string(x, )
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                    name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x






def get_fixlen_feature_names(feature_columns):
    features = build_input_features(feature_columns, include_varlen=False,include_fixlen=True)
    return list(features.keys())

def get_varlen_feature_names(feature_columns):
    features = build_input_features(feature_columns, include_varlen=True,include_fixlen=False)
    return list(features.keys())


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))

def build_input_features(feature_columns, include_varlen=True, mask_zero=True, prefix='',include_fixlen=True):
    input_features = OrderedDict()
    if include_fixlen:
        for fc in feature_columns:
            if isinstance(fc,SparseFeat):
                input_features[fc.name] = Input(
                    shape=(1,), name=prefix+fc.name, dtype=fc.dtype)
            elif isinstance(fc,DenseFeat):
                input_features[fc.name] = Input(
                    shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
                
                
            elif isinstance(fc,ImgDenseFeat):
                input_features[fc.name] = Input(shape=(224, 224, 3,), name=prefix + fc.name, dtype=fc.dtype)
                #Input(shape=(fc.length,), name=prefix + fc.name, dtype=fc.dtype)
                
                
                
                
    if include_varlen:
        for fc in feature_columns:
            if isinstance(fc,VarLenSparseFeat):
                input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + 'seq_' + fc.name,
                                                      dtype=fc.dtype)
        if not mask_zero:
            for fc in feature_columns:
                input_features[fc.name+"_seq_length"] = Input(shape=(
                    1,), name=prefix + 'seq_length_' + fc.name)
                input_features[fc.name+"_seq_max_length"] = fc.maxlen


    return input_features



    # embedding sparse feature (fix or variable length) into an dictionary using keras embedding function
def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, embedding_size,
                                                       embeddings_initializer=RandomNormal(
                                                       mean=0.0, stddev=init_std, seed=seed),
                                                       embeddings_regularizer=l2(l2_reg),
                                                       name=prefix + '_emb_'  + feat.name) for feat in sparse_feature_columns}
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                        embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(l2_reg),
                                                        name=prefix + '_seq_emb_' + feat.name,mask_zero=seq_mask_zero)
    return sparse_embedding
#--------------------------------------------------------------------------------------------------------------------------------------# apply the embedding operation on the feature columns
def create_embedding_matrix(feature_columns,l2_reg,init_std,seed,embedding_size, prefix="",seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed,
                                                 l2_reg, prefix=prefix + 'sparse',seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


    # two look up functions used to extract feature from dictionary to an list
def embedding_lookup(sparse_embedding_dict,sparse_input_dict,sparse_feature_columns,return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0  or feature_name in return_feat_list and fc.embedding:
            if fc.use_hash:
                lookup_idx = Hash(fc.dimension,mask_zero=(feature_name in mask_feat_list))(sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))
    return embedding_vec_list

def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.dimension, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict

  #final input to NN
def combined_dnn_input(sparse_embedding_list,dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(Concatenate(axis=-1)(sparse_embedding_list))
        dense_dnn_input = Flatten()(Concatenate(axis=-1)(dense_value_list))
        return Concatenate(axis=-1)([sparse_dnn_input,dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(Concatenate(axis=-1)(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(Concatenate(axis=-1)(dense_value_list))
    else:
        raise NotImplementedError