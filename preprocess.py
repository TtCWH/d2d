# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'
import pdb
import os

def data_id_name(id2e,id2r,dataset="FB15k",file="traindata"):
	lines=[]
	if os.path.exists("{}/{}_parse.txt".format(dataset,file)):
		print("{}/{}_parse.txt has been generated!".format(dataset,file))
		return
	print("generating {}/{}_parse.txt".format(dataset,file))
	with open("{}/{}.txt".format(dataset,file)) as f:
		for _ in f.readlines():
			_=_.split()
			v="-"
			if _[-1]=='0':
				v="+"
			lines.append(id2e[int(_[0])]+" "+id2e[int(_[2])]+" "+id2r[int(_[1])]+" "+v+'\n')
	with open("{}/{}_parse.txt".format(dataset,file),'a') as f:
		for _ in lines:
			f.write(_)

def name_id(dataset="FB15k",file="entity"): #transform entity or relation to index
	name2id={}
	id2name={}
	with open("{}/{}2id.txt".format(dataset,file)) as f:
		for _ in f.readlines():
			name2id[_.split()[0].strip()]=int(_.split()[1].strip())
			id2name[int(_.split()[1].strip())]=_.split()[0].strip()
	return name2id,id2name

def data_index(id2e,e2id,id2r,r2id,dataset="FB15k",file="train"): #transform dataset to index
	data={}
	inputE_index,output_index,inputR_index=[],[],[]
	if os.path.exists("{}/{}data.txt".format(dataset,file)):
		print("{} {}data has been generated!".format(dataset,file))
		data_id_name(id2e,id2r,dataset,"{}data".format(file))
		with open("{}/{}data.txt".format(dataset,file)) as f:
			for _ in f.readlines():
				_=_.split()
				h,r,t,v=int(_[0]),int(_[1]),int(_[2]),int(_[3])
				inputE_index.append([h,t])
				inputR_index.append([r])
				output_index.append(v)
		return inputE_index,inputR_index,output_index
	print("generating {} {}data".format(dataset,file))
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
		# print(length)	
	with open("{}/{}data.txt".format(dataset,file),'a') as f:
		for h,r,t in data.keys():
			f.write("{} {} {} {}\n".format(h,r,t,data[(h,r,t)]))
			inputE_index.append([h,t])
			inputR_index.append([r])
			output_index.append(data[(h,r,t)])
	data_id_name(id2e,id2r,dataset,"{}data".format(file))
	return inputE_index,inputR_index,output_index


def get_test_top(id2e,e2id,id2r,r2id,dataset="FB15k",file="test"):
	data={}
	inputE_index,output_index,inputR_index=[],[],[]
	if os.path.exists("{}/{}_top_data.txt".format(dataset,file)):
		print("{} {}_top_data has been generated!".format(dataset,file))
		data_id_name(id2e,id2r,dataset,"{}_top_data".format(file))
		with open("{}/{}data.txt".format(dataset,file)) as f:
			for _ in f.readlines():
				_=_.split()
				h,r,t,v=int(_[0]),int(_[1]),int(_[2]),int(_[3])
				inputE_index.append([h,t])
				inputR_index.append([r])
				output_index.append(v)
		return inputE_index,inputR_index,output_index
	print("generating {} {}_top_data".format(dataset,file))
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
		for h,r,t in data.keys():
			for temp_t in id2e.keys():
				try:
					a=data[(h,r,temp_t)]
				except:
					data[(h,r,temp_t)]=1
		# print(length)	
	with open("{}/{}_top_data.txt".format(dataset,file),'a') as f:
		for h,r,t in data.keys():
			f.write("{} {} {} {}\n".format(h,r,t,data[(h,r,t)]))
			inputE_index.append([h,t])
			inputR_index.append([r])
			output_index.append(data[(h,r,t)])
	data_id_name(id2e,id2r,dataset,"{}_top_data".format(file))
	return inputE_index,inputR_index,output_index



if __name__=="__main__":
	e2id,id2e=name_id()
	r2id,id2r=name_id(file="relation")
	data_index(id2e,e2id,id2r,r2id,dataset="FB15k",file="train")
	data_index(id2e,e2id,id2r,r2id,dataset="FB15k",file="test")