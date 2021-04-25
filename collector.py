import os
from pathlib import Path
from shutil import copy

clusters = ["5", "7", "10", "15", "20"]

souce_path = "slic"
search_path = "../resultk20"
color_model = "hsv"
source_images = os.listdir(souce_path)
for i in source_images:
    if os.path.isdir(Path(f"{souce_path}/{i}")):
        continue
    for c in clusters:
        cluster_path = f"{search_path}/{color_model}/{c}"
        cluster_images = os.listdir(cluster_path)
        cluster_image_index = cluster_images.index(i)
        copy(f"{cluster_path}/{cluster_images[cluster_image_index]}", f"clusterization/{color_model}/{c}")
