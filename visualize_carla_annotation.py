"""
This small script will show you how to read labels from the Carla dataset that you built. 
"""

import cv2
import numpy as np
from utils import Box, make_euler, get_position_ground
import os
import pandas as pd

if __name__ == "__main__":
   
    # Use same pitch and height to match reality
    to_world_from_camera = np.identity(4)
    pitch_matrix = make_euler(-27.39 * np.pi / 180, 0)
    to_world_from_camera[:3, :3] = pitch_matrix
    to_world_from_camera[:3, 3] = np.array([0, 8.5, 0])

    # (from CARLA)
    K = np.array([[770, 0.0, 640.0], [0.0, 770, 360.0], [0.0, 0.0, 1.0]])

    for img_path in sorted(os.listdir("carla_dataset/images")):
        img = cv2.imread(f"carla_dataset/images/{img_path}")

        label_idx = int(img_path[-8:-4])

        if label_idx <= 0:
            continue

        labels = pd.read_csv(
            f"carla_dataset/labels/image_{label_idx:04d}.txt"
        ).to_numpy()
        # print(labels)
        for label in labels:
            position_on_ground = get_position_ground(
                label[0], label[1], K, to_world_from_camera, img.shape[0]
            )

            box = Box(
                position_on_ground[0],
                position_on_ground[2],
                label[2],
                label[3],
                label[4],
            )
            # print(label)
            img = box.project(img, to_world_from_camera, K)

        cv2.imshow("Image", img)
        k = cv2.waitKey(0)

        if k == 27:
            break
