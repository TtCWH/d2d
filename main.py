# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import os
import pdb
import time
import numpy as np
import tensorflow as tf
from preprocess import data_index,name_id,get_test_top
from d2d_model import D2dmodel
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def test_model_top(e2id,r2id,id2e,id2r,model,session,epoch,flag="test on testdata for top_k"):
	print(flag)
	e_test,r_test,y_test=get_test_top(id2e,e2id,id2r,r2id,file="test")
	e_test=np.asarray(e_test,dtype="int32")
	r_test=np.asarray(r_test,dtype="int32")
	y_test=np.asarray(y_test,dtype="int32")
	# batch_tensor=np.zeros(y_test.shape,dtype="int32")
	predictions=model.prediction.eval(feed_dict={model.inputE:e_test,model.inputR:r_test},session=session)
	# loss=model.loss.eval(feed_dict={model.inputE:e_test,model.inputR:r_test,model.y_label:y_test})
	# predictions=np.argmax(predictions,1)
	top_10=np.argsort(predictions,0)[:10]
	top_3=np.argsort(predictions,0)[:3]
	top_1=np.argsort(predictions,0)[:1]
	ans_10,ans_3,ans_1=0,0,0
	for i in range(len(top_10)):
		if y_test[top_10[i][0]]==0:
			ans_10+=1
	for i in range(len(top_3)):
		if y_test[top_3[i][0]]==0:
			ans_3+=1
	for i in range(len(top_1)):
		if y_test[top_1[i][0]]==0:
			ans_1+=1
	# print("test loss:{}".format(loss))
	print("top_10 accuray:{}".format(float(ans_10)/10.0))
	print("top_3 accuray:{}".format(float(ans_3)/3.0))
	print("top_1 accuray:{}".format(float(ans_1)/1.0))
	with open("run_top.log",'a') as f:
		# f.write("test loss:{}".format(loss)+'\n')
		f.write("top_10 accuray:{}".format(float(ans_10)/10.0)+'\n')
		f.write("top_3 accuray:{}".format(float(ans_3)/3.0)+'\n')
		f.write("top_1 accuray:{}".format(float(ans_1)/1.0)+'\n')


def test_model(e2id,r2id,id2e,id2r,model,session,epoch,flag="test on testdata"):
	print(flag)
	e_test,r_test,y_test=data_index(id2e,e2id,id2r,r2id,file="test")
	e_test=np.asarray(e_test,dtype="int32")
	r_test=np.asarray(r_test,dtype="int32")
	y_test=np.asarray(y_test,dtype="int32")
	# batch_tensor=np.zeros(y_test.shape,dtype="int32")
	predictions=model.prediction.eval(feed_dict={model.inputE:e_test,model.inputR:r_test},session=session)
	loss=model.loss.eval(feed_dict={model.inputE:e_test,model.inputR:r_test,model.y_label:y_test})
	predictions=np.argmax(predictions,1)
	ans=0
	for i in range(len(predictions)):
		if predictions[i]==y_test[i]:
			ans+=1

	print("test loss:{}".format(loss))
	print("accuray:{}".format(float(ans)/float(len(predictions))))
	with open("run.log",'a') as f:
		f.write("test loss:{}".format(loss)+'\n')
		f.write("accuray:{}".format(float(ans)/float(len(predictions)))+'\n')


def train_model(epochs=100,batchsize=50):
	"""
	epochs:训练轮数
	"""
	e2id,id2e=name_id()
	r2id,id2r=name_id(file='relation')
	e_train,r_train,y_train=data_index(id2e,e2id,e2id,r2id)
	e_train=np.asarray(e_train,dtype="int32")
	r_train=np.asarray(r_train,dtype="int32")
	y_train=np.asarray(y_train,dtype="int32")

	with tf.Session() as sess:
		with tf.variable_scope("d2dmodel",reuse=None):
			m=D2dmodel(entity_num=e_train.shape[0],relation_num=r_train.shape[0])
		with tf.variable_scope("d2dmodel",reuse=True):
			m_test=D2dmodel(entity_num=e_train.shape[0],relation_num=r_train.shape[0],is_training=False,keep_prob=1.0)
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			print("epoch:{}".format(epoch))
			for e_data,r_data,y_data in get_train_batch(e_train,r_train,y_train,50):
				# pdb.set_trace()
				# batch_tensor=np.zeros(y_data.shape,dtype="int32")
				m.train_step.run(feed_dict={m.inputE:e_data,m.inputR:r_data,m.y_label:y_data})
				value=m.loss.eval(feed_dict={m.inputE:e_data,m.inputR:r_data,m.y_label:y_data})
				# print('loss: {}'.format(value))
			test_model(e2id,r2id,id2e,id2r,m_test,sess,epoch)
			test_model_top(e2id,r2id,id2e,id2r,model,session,epoch)


if __name__=="__main__":
	with open('run.log','a') as f:
		f.write("start time : "+time.strftime("%Y-%m-%d %H:%M:%S")+'\n')
	print(time.strftime("%Y-%m-%d %H:%M:%S"))
	train_model(batchsize=5000)