
## Kaggle Airbus Ship Detection Challenge : 21st solution

This project is for Kaggle competiton [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection), which can **help you quickly get a baseline solution**.

[![FVXIkF.md.jpg](https://s1.ax1x.com/2018/11/28/FVXIkF.md.jpg)](https://imgchr.com/i/FVXIkF)



## Related article

[Kaggle新手银牌（21st）：Airbus Ship Detection 卫星图像分割检测](https://zhuanlan.zhihu.com/p/48381892)

[用Mask R-CNN训练自己的COCO数据集（Detectron）](https://zhuanlan.zhihu.com/p/50127900)



## File strcture

    airbus                         
    ├─0_rle_to_coco                0、turn rle to coco
    │  └─pycococreatortools
    |
    ├─1_detectron_infer            1、files needed to be changed in detectron
    |  ├─dataset_catalog.py            # ./detectron/datasets/dataset_catalog.py
    │  ├─dummy_datasets.py             # ./detectron/datasets/dummy_datasets.py 
    │  └─infer_airbus.py               # ./tools/infer_simple.py    
    |
    ├─2_model                      2、model and trainning log
    │  ├─log                           log and visualization script
    │  └─model                         configure file and .pkl
    |
    └─3_submit                     3、generate your submission
       └─csv                           reference .csv file



## Steps

#### 1、Generate COCO standard dataset 

Run codes in [./0_rle_to_coco](https://github.com/pascal1129/airbus_ship_detection/tree/master/0_rle_to_coco). The dependence and guide has been written in [./0_rle_to_coco/README.md](https://github.com/pascal1129/airbus_ship_detection/blob/master/0_rle_to_coco/README.md)

![dataset annotation](https://s1.ax1x.com/2018/10/31/iWlN8A.png)



#### 2、Get Detectron environment

You can use my docker image, which is a little out of date, by the following command:

```
$ docker pull pascal1129/detectron:caffe2_cuda9_aliyun
```

Or you can build the official dockerfile:  [Detectron/docker/Dockerfile](https://github.com/facebookresearch/Detectron/blob/master/docker/Dockerfile), in order to get the latest docker image.



#### 3、Msodify the source code in detectron 

My codes are in the folder [./1_detectron_infer/](https://github.com/pascal1129/airbus_ship_detection/tree/master/1_detectron_infer), you can replace the origin files in detectron with my codes. 



#### 4、Change the configuration file and run

 Confirm the .yaml file in `./2_model/model` and start training. In addition, remember to use `|tee` command, so you can get the log file like [./2_model/log/20181103.log](https://github.com/pascal1129/airbus_ship_detection/blob/master/2_model/log/20181103.log). 



#### 5、visualization

Run [analyse_log.py](https://github.com/pascal1129/airbus_ship_detection/blob/master/2_model/analyse_log.py), then you can get the visualization picture.

![result](https://github.com/pascal1129/airbus_ship_detection/blob/master/2_model/log/20181103.png)


#### 6、Get the final submission

Run [get_final_csv.py](https://github.com/pascal1129/airbus_ship_detection/blob/master/3_submit/get_final_csv.py)

