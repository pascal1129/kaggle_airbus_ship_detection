import numpy as np
import pandas as pd
import os

dataset_dir = 'D:/Downloads/test'
csv_origin = './example.csv'
csv_unet   = './unet.csv'
csv_submit = './rle_submit.csv'

def generate_final_csv(df_with_ship):
	print("最终提交版本			:  %d instances,  %d images"  %(df_with_ship.shape[0], len(get_im_list(df_with_ship))))
	im_no_ship = get_im_no_ship(df_with_ship)
	# write dataframe into .csv file
	df_empty = pd.DataFrame({'ImageId':im_no_ship, 'EncodedPixels':get_empty_list(len(im_no_ship))})
	df_submit = pd.concat([df_with_ship, df_empty], sort=False)
	df_submit.drop(['area','confidence'], axis=1, inplace=True)
	df_submit.to_csv(csv_submit, index=False,sep=str(','))   # str(',') is needed
	print('Generate successfully!')

def get_im_no_ship(df_with_ship):
	im_all = os.listdir(dataset_dir)
	im_no_ship = list(set(im_all).difference(set(df_with_ship['ImageId'].tolist())))
	return im_no_ship

def get_empty_list(length):
    list_empty = []
    for _ in range(length):
        list_empty.append('')
    return list_empty

def get_im_list(df):
	df_with_ship = df[df['EncodedPixels'].notnull()]['ImageId']
	return list(set(df_with_ship))

if __name__ == '__main__':
	# 第一步：筛选detectron检测结果，主要根据阈值、面积
	df = pd.read_csv(csv_origin)
	print("Detectron原始结果	:  %d instances,  %d images"  %(df.shape[0], len(get_im_list(df))))
	df = df[ (df['area']>50) & (df['confidence']>=0.85) ]
	print("根据阈值筛选后		:  %d instances,  %d images"  %(df.shape[0], len(get_im_list(df))))

	# 第二步：融合unet模型
	df_unet = pd.read_csv(csv_unet)
	unet_no_ship_list = df_unet[df_unet['EncodedPixels'].isnull()]['ImageId'].tolist()
	df = df[~df['ImageId'].isin(unet_no_ship_list)]


	#---------------------------------------------#
	#	1. add images without ship
	#	2. drop column ['area','confidence']
	#	3. write dataframe into .csv
	#---------------------------------------------#
	generate_final_csv(df)