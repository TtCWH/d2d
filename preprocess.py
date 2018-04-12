# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'
import pdb
import os

def name_id(dataset="FB15k",file="entity"): #transform entity or relation to index
	name2id={}
	id2name={}
	with open("{}/{}2id.txt".format(dataset,file)) as f:
		for _ in f.readlines():
			name2id[_.split()[0].strip()]=int(_.split()[1].strip())
			id2name[int(_.split()[1].strip())]=_.split()[0].strip()
	return name2id,id2name

def data_index(e2id,r2id,dataset="FB15k",file="train"): #transform dataset to index
	data={}
	inputE_index,output_index,inputR_index=[],[],[]
	if os.path.exists("{}/{}data.txt".format(dataset,file)):
		with open("{}/{}data.txt".format(dataset,file)) as f:
			for _ in f.readlines():
				_=_.split()
				h,r,t,v=int(_[0]),int(_[1]),int(_[2]),int(_[3])
				inputE_index.append([h,t])
				inputR_index.append([r])
				output_index.append(v)
		return inputE_index,inputR_index,output_index
	with open("{}/{}.txt".format(dataset,file)) as f:
		lines=f.readlines()
		length=len(lines)
		for _ in lines:
			_=_.split()
			h=e2id[_[0].strip()]
			r=r2id[_[2].strip()]
			t=e2id[_[1].strip()]
			data[(h,r,t)]=0
		# pdb.set_trace()
		while length>0:
			for h,r,t in list(data.keys()):
				if length<=0:
					break
				for temp_t in e2id.values():
					if temp_t==h or temp_t==t:
						continue
					try:
						if data[h,r,temp_t]:
							continue
					except:
						try:
							if data[temp_t,r,h]:
								continue
						except:
							data[h,r,temp_t]=1
							length-=1
							if length<=0:
								break
		print(length)	
	with open("{}/{}data.txt".format(dataset,file),'a') as f:
		for h,r,t in data.keys():
			f.write("{} {} {} {}\n".format(h,r,t,data[(h,r,t)]))
			inputE_index.append([h,t])
			inputR_index.append([r])
			output_index.append(data[(h,r,t)])
	return inputE_index,inputR_index,output_index

if __name__=="__main__":
	e2id=name_id()[0]
	r2id=name_id(file="relation")[0]
	data_index(e2id,r2id,dataset="FB15k",file="train")