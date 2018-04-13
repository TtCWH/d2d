# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'
import os
from preprocess import name_id

def data_id_name(id2e,id2r,dataset="FB15k",file="traindata"):
	lines=[]
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

if __name__=="__main__":
	if os.path.exists("FB15k/traindata_parse.txt"):
		print("traindata has been parsed!")
	else:
		id2e=name_id()[1]
		id2r=name_id(file="relation")[1]
		data_id_name(id2e,id2r)
	if os.path.exists("FB15k/testdata_parse.txt"):
		print("testdata has been parsed!")
	else:
		id2e=name_id()[1]
		id2r=name_id(file="relation")[1]
		data_id_name(id2e,id2r,file="testdata")
