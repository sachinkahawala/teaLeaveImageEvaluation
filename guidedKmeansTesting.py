import cv2
import numpy as np
from math import sqrt as squareRoot
import time
import argparse
import numpy as np
from skimage import io, img_as_float
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def print_distortion_distance(cluster_prototypes, points_by_label, k):
    distances = np.zeros((k,))

    for k_i in range(k):
        if (points_by_label[k_i] is not None):
            distances[k_i] += np.linalg.norm(points_by_label[k_i] - cluster_prototypes[k_i], axis=1).sum()
        else:
            distances[k_i] = -1

    print('Distortion Distances:')
    print(distances)
def k_means_clustering(image, k, num_iterations):

    image_vectors = image.reshape(-1, image.shape[-1])
    # Create corresponding label array (Initialize with Label: -1)
    labels = np.full((image_vectors.shape[0],), -1)
    # Assign Initial Cluster Prototypes
    cluster_prototypes = np.random.rand(k, 3)

    # Iteration Loop
    for i in range(num_iterations):
        start=time.time()
        print('Iteration: ' + str(i + 1))
        points_by_label = [None for k_i in range(k)]

        # Label them via closest point
        for rgb_i, rgb in enumerate(image_vectors):
            # [rgb, rgb, rgb, rgb, ...]
            rgb_row = np.repeat(rgb, k).reshape(3, k).T

            # Find the Closest Label via L2 Norm
            closest_label = np.argmin(np.linalg.norm(rgb_row - cluster_prototypes, axis=1))
            labels[rgb_i] = closest_label

            if (points_by_label[closest_label] is None):
                points_by_label[closest_label] = []

            points_by_label[closest_label].append(rgb)

        # Optimize Cluster Prototypes (Center of Mass of Cluster)
        for k_i in range(k):
            if (points_by_label[k_i] is not None):
                new_cluster_prototype = np.asarray(points_by_label[k_i]).sum(axis=0) / len(points_by_label[k_i])
                cluster_prototypes[k_i] = new_cluster_prototype

        # Find Current Distortion Distances
        print_distortion_distance(cluster_prototypes, points_by_label, k)
        print(labels)
        print(time.time()-start)

    return (labels, cluster_prototypes)
img = cv2.imread('D:/Projects/FYP/image classification/results/4-K_means_clusterisation_applied.jpg')
k_means_clustering(img,4,4)
