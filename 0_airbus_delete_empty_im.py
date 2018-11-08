import os
import pandas as pd
import numpy as np
from PIL import Image

dataset_train = '../datasets/ships_train2018'
csv_train = 	'../datasets/train_ship_segmentations_v2.csv'

if __name__ == '__main__':
    # read_csv_file
    df = pd.read_csv(csv_train)
    print("Dataframe lines ： ",df.shape[0])

    # delete annotations without ship
    df = df.dropna(axis=0)
    num_of_ships = df.shape[0]
    print("Inastances      ： ",num_of_ships)

    # create an empty set to store images with ship
    images = set()
    for line in range(num_of_ships):
        if df.iloc[line,0] not in images:
            images.add(df.iloc[line,0])
    print("Images with ship： ",len(images))

    # Delete images without ship
    count = 0
    ims = os.listdir(dataset_train)
    for im in ims:
        if im not in images:
            os.remove(os.path.join(im_path, im))
            count += 1
    print('%d images is deleted.'%(count))