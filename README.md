
## Kaggle Airbus Ship Detection Challenge : 21st solution

This project is for Kaggle competiton [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).

It can help you quickly get a **baseline solution**, which is not bad.

![infer_example](https://github.com/pascal1129/kaggle_airbus_ship_detection/blob/master/images/infer_example.jpg)





## Related article

#### These guides are only in Chinese：

[Kaggle新手银牌（21st）：Airbus Ship Detection 卫星图像分割检测](https://zhuanlan.zhihu.com/p/48381892)

[用Mask R-CNN训练自己的COCO数据集（Detectron）](https://zhuanlan.zhihu.com/p/50127900)

[辅助操作指南：Docker使用、镜像制作、Demo运行...](https://github.com/pascal1129/public_notes)





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
    │  └─model                         configure file and .pkl (.pkl not be uploaded)
    |
    └─3_submit                     3、generate your submission
       └─csv                           reference .csv file





## Steps

#### 1. Generate COCO standard dataset 

Run codes in `./0_rle_to_coco`. The guide has been written in markdwon file `./0_rle_to_coco/README.md`

![dataset annotation](https://github.com/pascal1129/kaggle_airbus_ship_detection/blob/master/images/annotation.png)



#### 2. Get Detectron environment

My codes are based on [Detectron](https://github.com/facebookresearch/Detectron). So before using it, you need to install caffe2, which is quite troublesome. You can use my docker image, which is a little out of date, by the following command:

```
$ docker pull pascal1129/detectron:caffe2_cuda9_aliyun
```

In order to get the latest docker image, you can build the latest image with the official dockerfile: [Detectron/docker/Dockerfile](https://github.com/facebookresearch/Detectron/blob/master/docker/Dockerfile).



#### 3. Msodify the source code in detectron 

My codes are in the folder `./1_detectron_infer/`, you can replace the origin files in detectron with my codes. 

my code|origin code needed to be replaced
---------------------------------|--------------
dataset_catalog.py        | ./detectron/datasets/dataset_catalog.py
dummy_datasets.py     | ./detectron/datasets/dummy_datasets.py 
infer_airbus.py              | ./tools/infer_simple.py   




#### 4. Change the configuration file and run

 Confirm the .yaml file in `./2_model/model/` and start training. In addition, remember to use `|tee` command, so you can get the log file like [./2_model/log/20181103.log](https://github.com/pascal1129/airbus_ship_detection/blob/master/2_model/log/20181103.log)



#### 5. Visualization

Run `./2_model/analyse_log.py`, then you can get the visualization picture.

![result](https://github.com/pascal1129/airbus_ship_detection/blob/master/2_model/log/20181103.png)




#### 6. Get the final submission

Run `./3_submit/get_final_csv.py`.
