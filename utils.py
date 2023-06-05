"""
Define a bunch of utils class and functions related to geometry and opencv.
"""

import cv2
import numpy as np


def project(point_in_world, to_camera_from_world, K):
    """
    Given the camera intrinsic and extrinsic calibration matrices,
    projet a 3D point in world coordinates to the image
    """

    point_in_world = point_in_world.reshape(3, 1)
    point_in_camera = to_camera_from_world[:3, :3].dot(
        point_in_world
    ) + to_camera_from_world[:3, 3].reshape(3, 1)

    point_in_camera /= abs(point_in_camera[2])

    projection = K.dot(point_in_camera)
    projection = projection.flatten()

    return projection[:2]

def make_euler(angle, axis):
    """Define euler matrix with some angle around a specified axis"""

    if axis < 0:
        raise ValueError
    if axis > 3:
        raise ValueError

    if axis == 0:  # Pitch (axis = X)
        pitch_matrix = np.zeros((3, 3))
        pitch_matrix[0, 0] = 1
        pitch_matrix[1, 1] = np.cos(angle)
        pitch_matrix[1, 2] = np.sin(angle)
        pitch_matrix[2, 1] = -np.sin(angle)
        pitch_matrix[2, 2] = np.cos(angle)
        return pitch_matrix

    if axis == 1:  # Yaw (axis = Y)
        yaw_matrix = np.zeros((3, 3))
        yaw_matrix[0, 0] = np.cos(angle)
        yaw_matrix[0, 2] = np.sin(angle)
        yaw_matrix[1, 1] = 1
        yaw_matrix[2, 0] = -np.sin(angle)
        yaw_matrix[2, 2] = np.cos(angle)
        return yaw_matrix

    return False

def get_position_ground(x, y, K, to_world_from_camera, img_height):
    """
    Find position in world on the driveway (altitude = 0)
    by ray tracing a pixel point [x, y]
    """

    point_in_camera = np.linalg.inv(K).dot(
        np.array([x, img_height - y, 1]).reshape(3, 1)
    )
    point_in_camera = point_in_camera.reshape(3, 1)
    # Find intersection with Y = 0

    point_in_world = to_world_from_camera[:3, :3].dot(
        point_in_camera
    ) + to_world_from_camera[:3, 3].reshape(3, 1)

    camera_pos_in_world = to_world_from_camera[:3, 3].reshape(3, 1)

    direction = point_in_world - camera_pos_in_world
    direction /= np.linalg.norm(direction)

    lambda_ = -camera_pos_in_world[1] / direction[1]

    position_on_ground = (camera_pos_in_world + lambda_ * direction).flatten()

    return position_on_ground


def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def draw_frame_id(frame, frame_id):
    """Draw frame id to demonstrate performance"""
    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (frame.shape[1] - 150, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )

# Define a bunch of RGB colors (used in tracking)
COLORS = [
    (0, 0, 128),
    (0, 0, 255),
    (0, 128, 0),
    (0, 128, 128),
    (0, 128, 255),
    (0, 255, 0),
    (0, 255, 128),
    (0, 255, 255),
    (128, 0, 0),
    (128, 0, 128),
    (128, 0, 255),
    (128, 128, 0),
    (128, 128, 128),
    (128, 128, 255),
    (128, 255, 0),
    (128, 255, 128),
    (128, 255, 255),
    (255, 0, 0),
    (255, 0, 128),
    (255, 0, 255),
    (255, 128, 0),
    (255, 128, 128),
    (255, 128, 255),
    (255, 255, 0),
    (255, 255, 128),
    (255, 255, 255),
]


class Box:
    """
    Define 3D bounding box
    """

    def __init__(self, x, z, w, h, l):
        self.x = x
        self.z = z
        self.w = w
        self.h = h
        self.l = l

    def project(self, img, to_world_from_camera, K, color=(255, 0, 255)):
        """
        Project 3D bounding box to the image img
        """
        to_camera_from_world = np.linalg.inv(to_world_from_camera)

        left_bottom_back = project(
            np.array([self.x - self.w / 2, 0, self.z - self.l / 2]),
            to_camera_from_world,
            K,
        )
        left_bottom_front = project(
            np.array([self.x - self.w / 2, 0, self.z + self.l / 2]),
            to_camera_from_world,
            K,
        )
        left_top_back = project(
            np.array([self.x - self.w / 2, self.h, self.z - self.l / 2]),
            to_camera_from_world,
            K,
        )
        left_top_front = project(
            np.array([self.x - self.w / 2, self.h, self.z + self.l / 2]),
            to_camera_from_world,
            K,
        )
        right_bottom_back = project(
            np.array([self.x + self.w / 2, 0, self.z - self.l / 2]),
            to_camera_from_world,
            K,
        )
        right_bottom_front = project(
            np.array([self.x + self.w / 2, 0, self.z + self.l / 2]),
            to_camera_from_world,
            K,
        )
        right_top_back = project(
            np.array([self.x + self.w / 2, self.h, self.z - self.l / 2]),
            to_camera_from_world,
            K,
        )
        right_top_front = project(
            np.array([self.x + self.w / 2, self.h, self.z + self.l / 2]),
            to_camera_from_world,
            K,
        )

        left_bottom_back[1] = img.shape[0] - left_bottom_back[1]
        left_bottom_front[1] = img.shape[0] - left_bottom_front[1]
        left_top_back[1] = img.shape[0] - left_top_back[1]
        left_top_front[1] = img.shape[0] - left_top_front[1]
        right_bottom_back[1] = img.shape[0] - right_bottom_back[1]
        right_bottom_front[1] = img.shape[0] - right_bottom_front[1]
        right_top_back[1] = img.shape[0] - right_top_back[1]
        right_top_front[1] = img.shape[0] - right_top_front[1]

        left_bottom_back = tuple(left_bottom_back.astype(int))
        left_bottom_front = tuple(left_bottom_front.astype(int))
        left_top_back = tuple(left_top_back.astype(int))
        left_top_front = tuple(left_top_front.astype(int))
        right_bottom_back = tuple(right_bottom_back.astype(int))
        right_bottom_front = tuple(right_bottom_front.astype(int))
        right_top_back = tuple(right_top_back.astype(int))
        right_top_front = tuple(right_top_front.astype(int))

        # Draw back rectangle
        cv2.line(
            img,
            left_bottom_back,
            left_top_back,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            left_bottom_back,
            right_bottom_back,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            right_top_back,
            right_bottom_back,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            left_top_back,
            right_top_back,
            color=color,
            thickness=3,
        )

        # Draw front rectangle
        cv2.line(
            img,
            right_bottom_front,
            right_top_front,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            right_top_front,
            left_top_front,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            left_top_front,
            left_bottom_front,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            left_bottom_front,
            right_bottom_front,
            color=color,
            thickness=3,
        )

        # Draw Z lines
        cv2.line(
            img,
            right_bottom_back,
            right_bottom_front,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            right_top_front,
            right_top_back,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            left_top_back,
            left_top_front,
            color=color,
            thickness=3,
        )
        cv2.line(
            img,
            left_bottom_back,
            left_bottom_front,
            color=color,
            thickness=3,
        )

        image_overlay = img.copy()

        # Overlay red color over the whole bbox
        hull = cv2.convexHull(
            np.array(
                [
                    left_bottom_back,
                    left_bottom_front,
                    left_top_back,
                    left_top_front,
                    right_bottom_back,
                    right_bottom_front,
                    right_top_back,
                    right_top_front,
                ],
                dtype=int,
            )
        ).reshape(1, -1, 2)
        cv2.fillPoly(image_overlay, pts=hull, color=(0, 0, 255))

        alpha = 0.3
        img = cv2.addWeighted(img, 1 - alpha, image_overlay, alpha, 0)

        return img
