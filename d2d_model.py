# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class D2dmodel(object):
	"""docstring for d2dmodel"""
	def __init__(self, entity_num,relation_num,embedding_dim=100,epochs=100,
		is_training=True,init_lr=0.005,lstm_dim=3,keep_prob=0.75):
		"""
		epochs:训练轮数
		embedding_dim:lstm unit个数
		is_training:是否训练状态
		init_lr:初始learning rate
		lstm_dim:LSTM层维度
		entity_num:entity的个数
		relation_num:relation的个数
		"""
		num_steps=embedding_dim
		self._inputE=tf.placeholder(tf.int32,[None,2])
		self._inputR=tf.placeholder(tf.int32,[None,1])
		self._y_label=tf.placeholder(tf.int32,[None])

		#embedding layer
		with tf.device('/cpu:0'):
			self.e_embedding=e_embedding=tf.get_variable('entity_embedding',[entity_num,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			self.r_embedding=r_embedding=tf.get_variable('relation_embedding',[relation_num,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			self.e_embedding_output=e_embedding_output=tf.nn.embedding_lookup(e_embedding,self._inputE)
			self.r_embedding_output=r_embedding_output=tf.nn.embedding_lookup(r_embedding,self._inputR)
		if is_training and keep_prob<1.0:
			self.e_embedding_output=e_embedding_output=tf.nn.dropout(e_embedding_output,1-keep_prob)
			self.r_embedding_output=r_embedding_output=tf.nn.dropout(r_embedding_output,1-keep_prob)
		
		#biLSTM layer
		self.LSTM_inputE=LSTM_inputE=tf.split(tf.transpose(e_embedding_output,[0,2,1]),[1,1],2)
		self.LSTM_inputR=LSTM_inputR=tf.transpose(r_embedding_output,[0,2,1])
		self.LSTM_input=LSTM_input=tf.concat([LSTM_inputE[0],LSTM_inputR,LSTM_inputE[1]],2)
		self.forward_LSTM=forward_LSTM=LSTMCell(lstm_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
		self.backward_LSTM=backward_LSTM=LSTMCell(lstm_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
		self.biLSTM_output=biLSTM_output=tf.nn.bidirectional_dynamic_rnn(forward_LSTM,backward_LSTM,LSTM_input,sequence_length=self._y_label,dtype=tf.float32)[0]
		
		#softmax layer
		softmax_input=tf.concat(biLSTM_output,2)
		self.softmax_input=softmax_input=tf.reshape(softmax_input,[-1,softmax_input.get_shape().as_list()[1]*softmax_input.get_shape().as_list()[-1]])
		self.softmax_W=softmax_W=tf.get_variable('softmax_Weight',[2*num_steps*lstm_dim,2],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
		self.softmax_b=softmax_b=tf.get_variable('softmax_bias',[2],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
		logits=tf.matmul(softmax_input,softmax_W)+softmax_b
		y_pre=tf.nn.softmax(logits)
		# pdb.set_trace()
		self._predictions=y_pre
		self._loss=loss=tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._y_label,logits=logits)))

		if not is_training:
			return

		global_step=tf.Variable(0)
		learning_rate = tf.train.exponential_decay(init_lr, global_step, epochs*100, 0.98, staircase=True)

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

