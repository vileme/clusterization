from connected_components import  count_components
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import jaccard_score

metric = dict.fromkeys(['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks'])
for k in metric:
    metric[k] = []


def clusterize(image_path, color_model, k, gt_path, result_path):
    image_array = cv2.imread(image_path)
    image_array = cv2.resize(image_array, (512, 512))
    if color_model == "hsv":
        conversion = cv2.COLOR_BGR2HSV
        output = cv2.COLOR_HSV2BGR
    elif color_model == "lab":
        conversion = cv2.COLOR_BGR2LAB
        output = cv2.COLOR_LAB2BGR
    image_array = cv2.cvtColor(image_array, conversion)
    components, result_image = kmeans(image_array, k, gt_path, result_path, Path(image_path).name)
    cv2.imwrite(f'{result_path}/{Path(image_path).name}', cv2.cvtColor(result_image, output))
    with open(f'{result_path}/components.txt', 'a') as f:
        f.write(f'{Path(image_path).name} components = {len(components)}\n')


def find_good_clusters_ind(clusters_size, label, gt):
    good = []
    pixel_amount_in_cluster = np.array([0] * clusters_size.size)
    for i, val in enumerate(gt):
        if val == 0:
            continue
        pixel_amount_in_cluster[label[i]] += 1
    for i, val in enumerate(pixel_amount_in_cluster):
        if val > clusters_size[i] / 2:
            good.append(i)
    return good


def jaccard_index(good_clusters_ind, gt_vectorized, label):
    cluster_union = []
    ground_res = []
    for c in label:
        if c in good_clusters_ind:
            cluster_union.append(255)  # coz masks are white
            ground_res.append([0, 0, 255])
        else:
            cluster_union.append(0)
            ground_res.append([255, 255, 255])
    cluster_union = np.array(cluster_union)
    return np.array(ground_res), jaccard_score(gt_vectorized, cluster_union, pos_label=255)


def count_clusters_size(k, label):
    clusters_size = np.array([0] * k)
    for i, val in enumerate(label):
        clusters_size[val] += 1
    return clusters_size


def write_ground_res(ground_res, result_path, image_name, gt, intersec_over_union):
    mask_dir = f'{result_path}/result_masks'
    for i, g in enumerate(gt):
        for id, gd in enumerate(g):
            if np.all([x == 0 for x in gt[i, id]]):
                gt[i, id] = [255, 255, 255]
            else:
                gt[i, id] = [255, 0, 0]
    Path(mask_dir).mkdir(parents=True, exist_ok=True)
    merged = cv2.addWeighted(gt, 0.5, ground_res, 0.5, 0.0, dtype=cv2.CV_32F)
    if intersec_over_union == 0:
        image_name = image_name.replace(".jpg", f"_empty.jpg")
    result_path = f"{mask_dir}/{image_name}"
    cv2.imwrite(result_path, merged)


def validate(label, gtpath, k, result_path, image_name):
    label = label.flatten()
    assert (Path(gtpath).name.replace('_attribute', "") == image_name.replace(".jpg", ""))
    classes = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']
    clusters_size = count_clusters_size(k, label)
    for c in classes:
        gt = cv2.imread(f'{gtpath}_{c}.png')
        if np.all([x == [0, 0, 0] for x in gt]):  # ignore empty mask
            continue
        gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_vectorized = gt.reshape((-1, 3))
        gt_vectorized = np.delete(gt_vectorized, [1, 2], axis=1).flatten()
        good_clusters_ind = find_good_clusters_ind(clusters_size, label, gt_vectorized)
        ground_res, intersec_over_union = jaccard_index(good_clusters_ind, gt_vectorized, label)
        write_ground_res(ground_res.reshape(gt.shape), result_path, image_name.replace(".jpg", f"_{c}_gt.jpg"), gt,
                         intersec_over_union)
        with open(f'{result_path}/metrics.txt', 'a') as f:
            f.write(f'{image_name.replace(".jpg", "")} class = {c.ljust(20)} = {intersec_over_union} \n')
        metric[c].append(intersec_over_union)


def coordinate_img(img):
    width = img.shape[1]
    height = img.shape[0]
    colors = img.shape[2]
    a = np.full((height, width, colors + 2), 0.0)
    for h in range (height):
        for w in range (width):
            new = list(img[h,w])
            new.append(h)
            new.append(w)
            a[h,w] = new
    return a

def scale(a, b, img, max, min):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if k < 3:
                    continue
                img[i][j][k] = ((b - a) * (img[i][j][k] - 0)) / (max - min) + a


def kmeans(image, k, gt, result_path, image_name):
    coordinated = coordinate_img(image)
    # scale(0, coordinated.shape[0] / 3, coordinated, coordinated.shape[0], 0)
    vectorized = coordinated.reshape((-1, 5))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    validate(label, gt, k, result_path, image_name)
    components = count_components(label.reshape(image.shape[0], image.shape[1]))
    center = np.uint8(center)
    res = center[label.flatten()]
    res = np.delete(res, [3,4], axis = 1)
    result_image = res.reshape(image.shape)
    return components, result_image
