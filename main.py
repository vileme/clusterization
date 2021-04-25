import os
import shutil
from pathlib import Path
from statistics import mean
from random import shuffle
from linear_clusterize import clusterize, metric

train_data_path = r"test_data/ISIC2018_Task1-2_Training_Input"
train_gt_data_path = r"test_data/ISIC2018_Task2_Training_GroundTruth_v3"
train_images = os.listdir(train_data_path)
shuffle(train_images)
train_images = train_images[:300]
result_path_parent = 'visualization/linear_nonscale'
clusters = [5, 7, 10, 15, 20]
source_path = "visualization/slic"
source_images = os.listdir(source_path)
# if Path(result_path_parent).exists():
#     shutil.rmtree(Path(result_path_parent))

def process_data(color_model):
    for k in clusters:
        assert color_model == "hsv" or color_model == "lab"
        result_dir = f'{result_path_parent}/{color_model}/{k}'
        Path(result_dir).mkdir(parents=True, exist_ok=False)
        # for imgPath in train_images:
        #     print(imgPath)
        #     clusterize(f'{train_data_path}/{imgPath}', color_model, k,
        #                f'{train_gt_data_path}/{imgPath.replace(".jpg", "_attribute")}', result_dir)
        for imgPath in source_images:
            print(imgPath)
            if os.path.isdir(imgPath) or imgPath == "metrics.txt" or imgPath == "hist.png":
                continue

            clusterize(f'{train_data_path}/{imgPath}', color_model, k,
                       f'{train_gt_data_path}/{imgPath.replace(".jpg", "_attribute")}', result_dir)

        with open(f'{result_dir}/average.txt', 'a') as f:
            for criteria, v in metric.items():
                if len(v) == 0: continue
                avg = mean(v)
                f.write(f'average of class {criteria} = {avg}\n')
        for _, v in metric.items():
            v.clear()



# process_data("hsv")
process_data("lab")
