import math
import os
import sys
from pathlib import Path
from statistics import mean

import cv2
import numpy
import numpy as np
from skimage import io, color
from skimage.transform import resize
from sklearn.metrics import jaccard_score


def make_superPixel(h, w, img):
    return SuperPixels(h, w, img[h, w][0], img[h, w][1], img[h, w][2])


def initial_cluster_center(S, img, img_h, img_w, clusters):
    h = S // 2
    w = S // 2
    while h < img_h:
        while w < img_w:
            clusters.append(make_superPixel(h, w, img))
            w += S
        w = S // 2
        h += S
    return clusters


def calc_gradient(h, w, img, img_w, img_h):
    if w + 1 >= img_w:
        w = img_w - 2
    if h + 1 >= img_h:
        h = img_h - 2
    grad = img[h + 1, w + 1][0] - img[h, w][0] + img[h + 1, w + 1][1] - img[h, w][1] + img[h + 1, w + 1][2] - img[h, w][
        2]
    return grad


def reassign_cluster_center_acc_to_grad(clusters, img):
    for c in clusters:
        cluster_gradient = calc_gradient(c.h, c.w, img, img_w, img_h)
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                H = c.h + dh
                W = c.w + dw
                new_gradient = calc_gradient(H, W, img, img_w, img_h)
                if new_gradient < cluster_gradient:
                    c.update(H, W, img[H, W][0], img[H, W][1], img[H, W][2])
                    c_gradient = new_gradient


def assign_pixels_to_cluster(clusters, S, img, img_h, img_w, tag, dis):
    for c in clusters:
        for h in range(c.h - 2 * S, c.h + 2 * S):
            if h < 0 or h >= img_h: continue
            for w in range(c.w - 2 * S, c.w + 2 * S):
                if w < 0 or w >= img_w: continue
                l, a, b = img[h, w]
                Dc = math.sqrt(math.pow(l - c.l, 2) + math.pow(a - c.a, 2) + math.pow(b - c.b, 2))
                Ds = math.sqrt(math.pow(h - c.h, 2) + math.pow(w - c.w, 2))
                D = math.sqrt(math.pow(Dc / m, 2) + math.pow(Ds / S, 2))
                if D < dis[h, w]:
                    if (h, w) not in tag:
                        tag[(h, w)] = c
                        c.pixels.append((h, w))
                    else:
                        tag[(h, w)].pixels.remove((h, w))
                        tag[(h, w)] = c
                        c.pixels.append((h, w))
                    dis[h, w] = D


def update_cluster_mean(clusters):
    for c in clusters:
        sum_h = sum_w = number = 0
        # print("c.pixels",c.pixels)
        for p in c.pixels:
            sum_h += p[0]
            sum_w += p[1]
            number += 1
            H = sum_h // number
            W = sum_w // number
            c.update(H, W, img[H, W][0], img[H, W][1], img[H, W][2])


def find_good_clusters_ind(clusters, gt):
    good = []
    pixel_amount_in_cluster = np.array([0] * len(clusters))
    for i, c in enumerate(clusters):
        for p in c.pixels:
            pixel_index = p[0] * img_h + p[1]
            if (gt[pixel_index] == 0):
                continue
            pixel_amount_in_cluster[i] += 1
    for i, val in enumerate(pixel_amount_in_cluster):
        if val > len(clusters[i].pixels) / 2:
            good.append(i)
    return good


def jaccard_index(good_clusters_ind, gt_vectorized, clusters):
    if not good_clusters_ind:
        return 0
    cluster_union = [0] * img_h * img_w
    for i, c in enumerate(clusters):
        if i in good_clusters_ind:
            for p in c.pixels:
                pixel_index = p[0] * img_h + p[1]
                cluster_union[pixel_index] = 255
    cluster_union = np.array(cluster_union)
    return jaccard_score(gt_vectorized, cluster_union, pos_label=255)


# def validate_cluster(clusters, imgPath):
#     numpy.set_printoptions(threshold=sys.maxsize)
#     classes = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']
#     good = []
#     for c in classes:
#         gt_path = imgPath.replace(".jpg", "_attribute_" + c + ".png")
#         gt_img = np.array(cv2.imread(f'{train_gt_data_path}/{gt_path}'))
#         if np.all([x == [0, 0, 0] for x in gt_img]):  # ignore empty mask
#             continue
#         gt_vectorized = gt_img.reshape((-1, 3))
#         gt_vectorized = np.delete(gt_vectorized, [1, 2], axis=1).flatten()
#         good = find_good_clusters_ind(clusters, gt_vectorized)
#         intersec_over_union = jaccard_index(good, gt_vectorized, clusters)
#         with open(f'results/metrics.txt', 'a') as f:
#             f.write(f'{imgPath.replace(".jpg", "")} class = {c.ljust(20)} = {intersec_over_union} \n')
#         metric[c].append(intersec_over_union)
#         mask = np.copy(img)
#         for i, cl in enumerate(clusters):
#             for p in cl.pixels:
#                 if (i in good):
#                     mask[p[0], p[1]][0] = 100
#                     mask[p[0], p[1]][1] = 0
#                     mask[p[0], p[1]][2] = 0
#                 else:
#                     mask[p[0], p[1]][0] = 0
#                     mask[p[0], p[1]][1] = 0
#                     mask[p[0], p[1]][2] = 0
#         lab2rgb(mask, True, gt_path)
#     return good


def slic(S, img, img_h, img_w, clusters, tag, dis, imgPath):
    clusters = initial_cluster_center(S, img, img_h, img_w, clusters)
    reassign_cluster_center_acc_to_grad(clusters, img)
    for i in range(10):
        print(i)
        assign_pixels_to_cluster(clusters, S, img, img_h, img_w, tag, dis)
        update_cluster_mean(clusters)
        if i == 9:
            # validate_cluster(clusters, imgPath)
            image = np.copy(img)
            for c in clusters:
                for p in c.pixels:
                    image[p[0], p[1]][0] = c.l
                    image[p[0], p[1]][1] = c.a
                    image[p[0], p[1]][2] = c.b
            lab2rgb(image, False)
    return clusters


def lab2rgb(lab_arr, mask, gt_path=None):
    rgb_arr = color.lab2rgb(lab_arr)
    if (mask):
        io.imsave(f'{result_path}/masks/{gt_path}', rgb_arr)
    else:
        io.imsave(f'{result_path}/{imgPath}', rgb_arr)


class SuperPixels(object):

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b


train_data_path = r"HAM10000_images_part_1"
# train_gt_data_path = r"gt"
result_path = "ham_results"
train_images = os.listdir(train_data_path)
result_images = os.listdir(result_path)
# metric = dict.fromkeys(['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks'])
# for k in metric:
#     metric[k] = []
count = 0
for imgPath in train_images:
    print(f"{count} / {len(train_images)}")
    count += 1
    if imgPath in result_images:
        continue
    rgb = io.imread(f'{train_data_path}/{imgPath}', plugin='matplotlib')
    resize(rgb, (512, 512))
    img = color.rgb2lab(rgb)

    k = 300  # Number of Super pixels
    m = 20  # Constant for normalizing the color proximity, range of m = [1,40]

    img_h = img.shape[0]  # Image Height
    img_w = img.shape[1]  # Image Width

    N = img_h * img_w  # Total number of pixels in the image
    S = int(math.sqrt(N / k))  # average size of each superpixel

    clusters = []
    tag = {}
        # initialize the distance between pixels and cluster center as infinity
    dis = np.full((img_h, img_w), np.inf)

    cluster = slic(S, img, img_h, img_w, clusters, tag, dis, imgPath)
    # with open(f'results/average.txt', 'a') as f:
    #     for criteria, v in metric.items():
    #         if len(v) == 0: continue
    #         avg = mean(v)
    #         f.write(f'average of class {criteria} = {avg}\n')
