# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import os
import pdb
import numpy as np
import tensorflow as tf
from preprocess import data_index,name_id
from tensorflow.contrib.rnn import LSTMCell
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class D2dmodel(object):
	"""docstring for d2dmodel"""
	def __init__(self, entity_num,relation_num,embedding_dim=100,epochs=100,
		is_training=True,init_lr=0.001,lstm_dim=3,keep_prob=0.75):
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
			e_embedding=tf.get_variable('entity_embedding',[entity_num,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			r_embedding=tf.get_variable('relation_embedding',[relation_num,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			e_embedding_output=tf.nn.embedding_lookup(e_embedding,self._inputE)
			r_embedding_output=tf.nn.embedding_lookup(r_embedding,self._inputR)
		if is_training and keep_prob<1.0:
			e_embedding_output=tf.nn.dropout(e_embedding_output,1-keep_prob)
			r_embedding_output=tf.nn.dropout(r_embedding_output,1-keep_prob)
		
		#LSTM layer
		LSTM_inputE=tf.split(tf.transpose(e_embedding_output,[0,2,1]),[1,1],2)
		LSTM_inputR=tf.transpose(r_embedding_output,[0,2,1])
		LSTM_input=tf.concat([LSTM_inputE[0],LSTM_inputR,LSTM_inputE[1]],2)
		LSTM=LSTMCell(lstm_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
		LSTM_output=tf.nn.dynamic_rnn(LSTM,LSTM_input,dtype=tf.float32)[0]
		
		#softmax layer
		softmax_input=tf.reshape(LSTM_output,[-1,LSTM_output.get_shape().as_list()[1]*LSTM_output.get_shape().as_list()[-1]])
		softmax_W=tf.get_variable('softmax_Weight',[num_steps*lstm_dim,2],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
		softmax_b=tf.get_variable('softmax_bias',[2],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
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

		

def get_train_batch(inputE,inputR,inputY,batchsize,shuffle=True):
	"""
	inputE:实体对
	inputR:实体对对应的relation
	inputY:三元组的score
	batchsize:batch大小
	shuffle:打乱数据集
	"""
	assert len(inputE) == len(inputY)
	assert len(inputR) == len(inputY)
	indices=np.arange(len(inputY))
	if shuffle:
		np.random.shuffle(indices)
	for start_index in range(0,len(inputY)-batchsize+1,batchsize):
		sub_list=indices[start_index:start_index+batchsize]
		e=np.zeros((batchsize,2)).astype('int32')
		r=np.zeros((batchsize,1)).astype('int32')
		y=np.zeros((batchsize)).astype('int32')
		for i,index in enumerate(sub_list):
			e[i,]=inputE[index]
			r[i,]=inputR[index]
			y[i]=inputY[index]
		yield e,r,y


def test_model(e2id,r2id,id2e,id2r,model,session,epoch,flag="test on testdata"):
	print(flag)
	e_test,r_test,y_test=data_index(e2id,r2id,file="test")
	e_test=np.asarray(e_test,dtype="int32")
	r_test=np.asarray(r_test,dtype="int32")
	y_test=np.asarray(y_test,dtype="int32")

	predictions=model.prediction.eval(feed_dict={model.inputE:e_test,model.inputR:r_test,model.y_label:y_test})
	loss=model.loss.eval(feed_dict={model.inputE:e_test,model.inputR:r_test,model.y_label:y_test})
	predictions=tf.argmax(predictions,1)
	ans=0
	for i in range(len(predictions)):
		if predictions[i]==y_test[i]:
			ans+=1

	print("test loss:{}".format(loss))
	print("accuray:{}".format(float(ans)/float(len(predictions))))


def train_model(epochs=100,batchsize=50):
	"""
	epochs:训练轮数
	"""
	e2id,id2e=name_id()
	r2id,id2r=name_id(file='relation')[0]
	e_train,r_train,y_train=data_index(e2id,r2id)
	e_train=np.asarray(e_train,dtype="int32")
	r_train=np.asarray(r_train,dtype="int32")
	y_train=np.asarray(y_train,dtype="int32")

	with tf.Session() as sess:
		with tf.variable_scope("d2dmodel",reuse=None):
			m=D2dmodel(entity_num=e_train.shape[0],relation_num=r_train.shape[0])
		with tf.variable_scope("d2dmodel",reuse=True):
			m_test=D2dmodel(entity_num=e_train.shape[0],relation_num=r_train.shape[0],is_training=False)
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			print("epoch:{}".format(epoch))
			for e_data,r_data,y_data in get_train_batch(e_train,r_train,y_train,50):
				# pdb.set_trace()
				m.train_step.run(feed_dict={m.inputE:e_data,m.inputR:r_data,m.y_label:y_data})
				value=m.loss.eval(feed_dict={m.inputE:e_data,m.inputR:r_data,m.y_label:y_data})
				print('loss: {}'.format(value))
			test_model(m_test,sess,epoch)


if __name__=="__main__":
	train_model(batchsize=5000)