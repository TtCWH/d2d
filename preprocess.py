# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

def name_id(dataset="FB15k",file="entity"): #transform entity or relation to index
	name2id={}
	with open("{}/{}2id.txt".format(dataset,entity)) as f:
		for _ in f.readlines():
			entity2id[_.split()[0].strip()]=int(_.split()[1].strip())
	return name2id

def data_index(dataset="FB15k",file="train",e2id,r2id): #transform dataset to index
	inputE_index,output_index,inputR_index=[],[],[]
	with open("{}/{}.txt".format(dataset,file)) as f:
		for _ in f.readlines():
			_=_.split()
			inputE_index.append([e2id[_[0].strip()],e2id[_[1].strip()]])
			inputR_index.append([r2id[_[2].strip()]])
			output_index.append([1,0])
	return inputE_index,inputR_index,output_index

if __name__=="__main__":
	pass