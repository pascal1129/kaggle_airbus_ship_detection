import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

if __name__ == '__main__':
	iters = []	
	all_loss = []
	all_loss_bbox = []
	all_loss_mask = []
	all_accuracy_cls = []
	all_time = []
	all_lr =[]

	log_file = './log/20181103.log'

	with open(log_file) as f:
		for line in f:
			if ".yaml" in line:
				backbone = line.split('.yaml')[0].split('/')[-1]
				break
		for line in f:
			if "json_stats" not in line:
				continue
			accuracy_cls = line.split("\"accuracy_cls\": \"")[1].split("\"")[0]
			iteration = line.split("\"iter\": ")[1].split(",")[0]
			loss = line.split("\"loss\": \"")[1].split("\"")[0]
			loss_bbox = line.split("\"loss_bbox\": \"")[1].split("\"")[0]
			loss_mask = line.split("\"loss_mask\": \"")[1].split("\"")[0]
			time = line.split("\"time\": \"")[1].split("\"")[0]
			lr = line.split("\"lr\": \"")[1].split("\"")[0]

			accuracy_cls = float(accuracy_cls)
			iteration = int(iteration)
			loss = float(loss)
			loss_bbox = float(loss_bbox)
			loss_mask = float(loss_mask)
			time = float(time)
			lr = float(lr)
			print((accuracy_cls),iteration,loss,loss_bbox,loss_mask,time,lr)
			
			all_accuracy_cls.append(accuracy_cls)
			iters.append(iteration)
			all_loss.append(loss)
			all_loss_bbox.append(loss_bbox)
			all_loss_mask.append(loss_mask)
			all_time.append(time)
			all_lr.append(lr)

	fig = plt.figure(figsize=(8,6))
	ax2 = fig.add_subplot(111)
	ax1 = ax2.twinx()


	ax1.plot(iters, all_loss, color='red', label='loss')
	ax1.plot(iters, all_loss_bbox, color='purple', label='loss_bbox')
	ax1.plot(iters, all_loss_mask, color='blue', label='loss_mask')
	ax1.plot(iters, all_accuracy_cls, color='orange', label='accuracy_cls')
	ax1.plot(iters, all_time, color='olive', label='time/20iters')
	#设置坐标轴范围
	ax1.set_xlim((0,iters[-1]))
	ax1.set_ylim((0,1))

	# 设置坐标轴、图片名称
	ax1.set_xlabel('iters')
	log_name = log_file.split('.log')[0].split('/')[-1]
	ax1.set_title(log_name + ': ' +backbone)

	
	ax2.plot(iters, all_lr, color='black', label='lr')

	ax2.set_ylim([0, max(all_lr)*1.1])
	ax1.legend(loc='upper right')
	ax2.legend(loc='center right')


	ax1.set_ylabel('loss and accuracy')
	ax2.set_ylabel('learning rate')

	plt.savefig('./log/'+ log_name + '.png')
	plt.show()
