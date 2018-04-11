# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'
import pdb
import numpy as np
import tensorflow as tf
from preprocess import data_index,name_id
from tensorflow.contrib.rnn import LSTMCell

class Config:
	"""Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    dataset="FB15k"


class D2dmodel(object):
	"""docstring for d2dmodel"""
	def __init__(self, embedding_dim=100,epochs=100,
		is_training=True,init_lr=0.001,lstm_dim=3,entity_num,relation_num,keep_prob=0.75):
		"""
		embedding_dim:lstm unit个数
		epochs:训练轮数
		is_training:是否训练状态
		init_lr:初始learning rate
		lstm_dim:LSTM层维度
		entity_num:entity的个数
		relation_num:relation的个数
		"""
		num_steps=embedding_dim
		self._inputE=tf.placeholder(tf.int32,[None,2])
		self._inputR=tf.placeholder(tf.int32,[None,1])
		self._y_label=tf.placeholder(tf.float32,[None,2])

		#embedding layer
		with tf.device('/cpu:0'):
			e_embedding=tf.get_variable('entity_embedding',[entity_num,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			r_embedding=tf.get_variable('relation_embedding',[relation_num,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			e_embedding_output=tf.nn.embedding_lookup(e_embedding,self._inputE)
			r_embedding_output=tf.nn.embedding_lookup(r_embedding,self._inputR)
		if is_training and keep_prob<1.0:
			e_embedding_output=tf.nn.dropout(e_embedding_output,1-keep_prob)
			r_embedding_output=tf.nn.dropout(r_embedding_output,1-keep_prob)

		#LSTM layer
		LSTM_inputE=tf.split(tf.transpose(e_embedding_output,[0,2,1]),[1,1],1)
		LSTM_inputR=tf.transpose(r_embedding_output,[0,2,1])
		LSTM_input=tf.concat([LSTM_inputE[0],LSTM_inputR,LSTM_inputE[1]],1)
		LSTM=LSTMCell(lstm_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
		LSTM_output=tf.nn.dynamic_rnn(LSTM,LSTM_input,dtype=tf.float32)[0]

		#softmax layer
		softmax_input=tf.reshape(LSTM_output,[LSTM_output.shape[0],-1])
		softmax_W=tf.get_variable('softmax_Weight',[num_steps*lstm_dim,2],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
		softmax_b=tf.get_variable('softmax_bias',[2],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
		logits=tf.matmul(softmax_input,softmax_W)+softmax_b
		y_pre=tf.nn.softmax(logits)

		self._predictions=y_pre
		self._loss=loss=tf.losses.softmax_cross_entropy(self._y_label,logits)

		if not is_training:
			return

		global_step=tf.Variable(0)
        learning_rate = tf.train.exponential_decay(init_lr, global_step, npochos*100, 0.98, staircase=True)

		optimizer=tf.train.RMSPropOptimizer(learning_rate)
		self._train_op=optimizer.minimize(loss)

		@property
		def prediction(self):
			return self._predictions

		@property
		def train_step(self):
			return self._train_op

		@property
		def loss(self):
			return self._loss

		@property
		def inputE(self):
			return self._inputE

		@property
		def inputR(self):
			return self._inputR

		@property
		def y_label(self):
			return self._y_label

		

def get_train_batch(inputE,inputR,inputY,batchsize,shuffle=True):
	assert len(inputE) == len(inputY)
	assert len(inputR) == len(inputY)
	indices=np.arange(len(inputY))
	if shuffle:
		np.random.shuffle(indices)
	for start_index in range(0,len(inputY)-batchsize+1,batchsize):
		sub_list=indices[start_index:start_index+batchsize]
		e=np.zeros((batchsize,2)).astype('int32')
		r=np.zeros((batchsize,1)).astype('int32')
		y=np.zeros((batchsize,2)).astype('int32')
		for i,index in enumerate(sub_list):
			e[i,]=inputE[index]
			r[i,]=inputR[index]
			y[i,]=inputY[index]
		yield e,r,y



def train_model():
	e2id=name_id()
	r2id=name_id(file='relation')
	e_train,r_train,y_train=data_index()
