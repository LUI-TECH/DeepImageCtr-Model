import os
import copy
def Image_Procession(directory):
	directory_list = list()
	img_dict={}
	for root, dirs, files in os.walk(directory, topdown=True):
		for name in dirs:
			directory_list.append(os.path.join(root, name))
	for direction in directory_list:
		for root,dirs,files in os.walk(direction):
			img_dict[root.replace(directory,'')]=os.path.join(root, files[0])
	print("Total Number of different items:",len(img_dict))
	return img_dict


def Inputs_preProcess(img_dict,store):
	target=[]
	gender=[]
	uid=[]
	age=[]
	province=[]
	grade=[]
	interest=[]
	item_id=[]
	item_category=[]
	ad_pos=[]
	pool_id=[] 
	img_list=[]  
	for i in store:
		if i[0] in img_dict:
			new=store[i]
			target.append(new[1])
			uid.append(new[2])
			gender.append(new[3])
			age.append(new[4])
			province.append(new[5])
			grade.append(new[6])
			interest.append(new[7])
			item_id.append(new[8])
			item_category.append(new[9])
			ad_pos.append(new[10])
			pool_id.append(new[13])
			img_list.append(img_dict[i[0]])
	print("Total number of dataset for training: ",len(target))
	return target,uid,gender,age,province,grade,interest,item_id,item_category,ad_pos,pool_id,img_list


def read_ID(directory):
	f = open(directory,"r",encoding="utf-8")
	store={}
	for index,j in enumerate(f):
		i=copy.deepcopy(j)
		new=i.split(',',-1)
		item_id=new[8]
		uid=new[2]
		if new[2]!='0':
			store[(item_id,uid)]=new
	return store
