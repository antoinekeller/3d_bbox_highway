"""
This script demonstrates how you can determine camera pose from a single image
by locating lane lines. For this, you need to provide the instrinsic calibration matrix
and locate the distance between lines (check google maps for that).
"""

import cv2
import numpy as np
from utils import Box, make_euler, project, get_position_ground

class Line:
    """
    Line helper to project them on an image
    the direction if of course [0, 0, 1] since the lanes are parallel and along Z
    """
    def __init__(self, x):
        self.orig = np.array([x, 0, 0])
        self.dir = np.array([0, 0, 1])

    def project(self, to_world_from_camera, K):
        """Project a line (returns two points)"""
        to_camera_from_world = np.linalg.inv(to_world_from_camera)

        projection = project(self.orig, to_camera_from_world, K).flatten()

        far = self.orig + 1000 * self.dir
        projection_far = project(far, to_camera_from_world, K).flatten()

        return projection, projection_far

def find_translation_from_points(K, point1, point2, point3, x_positions):
    """
    From points of each line at the bottom of the image, estimate the camera translation.
    See the README.md for wider expanations
    """
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    A = np.zeros((6, 3))
    A[0, 0] = f
    A[1, 1] = f
    A[2, 0] = f
    A[3, 1] = f
    A[4, 0] = f
    A[5, 1] = f

    A[0, 2] = cx - point1[0]
    A[1, 2] = cy - point1[1]
    A[2, 2] = cx - point2[0]
    A[3, 2] = cy - point2[1]
    A[4, 2] = cx - point3[0]
    A[5, 2] = cy - point3[1]

    b = np.zeros((6, 1))
    b[0, 0] = -f * x_positions[0]
    b[2, 0] = -f * x_positions[1]
    b[4, 0] = -f * x_positions[2]

    ls = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    ls = ls.flatten()

    return ls

def draw_line(img, point1, point2):
    point1_cv = np.array([point1[0], img.shape[0] - point1[1]])
    point2_cv = np.array([point2[0], img.shape[0] - point2[1]])

    cv2.line(
        img,
        tuple(point1_cv.astype(int)),
        tuple(point2_cv.astype(int)),
        color=(0, 255, 0),
        thickness=3,
    )

def raycast(event, x, y, flags, params):
    """
    Raycast a bounding box with the mouse position.
    And draw it of curse!
    """

    img = params[0]
    img_with_lines = img.copy()

    K = params[1]
    to_world_from_camera = params[2]

    position_on_ground = get_position_ground(x, y, K, to_world_from_camera, img.shape[0])

    box = Box(position_on_ground[0], position_on_ground[2], w=1.6, h=1.6, l=4.5)
    img_with_lines = box.project(img_with_lines, to_world_from_camera, K)

    cv2.imshow("Image with car", img_with_lines)

    return img_with_lines


if __name__ == "__main__":
    img = cv2.imread("periph.jpg")

    # Check google maps to have the correct world coordinates of the lanes
    LINE_1_X = -0.95 - 2 * 3.45
    LINE_2_X = -0.95 - 3.45
    LINE_3_X = 0.95 + 3.45
    LINE_4_X = 0.95 + 2 * 3.45

    # pixel coordinates of the same lanes at the bottom
    BOTTOM_LINE_1 = 20
    BOTTOM_LINE_2 = 291
    BOTTOM_LINE_3 = 988
    BOTTOM_LINE_4 = 1260

    # pixel coordinates of the same lanes at the top
    TOP_LINE_1 = 601
    TOP_LINE_2 = 613
    TOP_LINE_3 = 650
    TOP_LINE_4 = 664

    h, w = img.shape[:2]

    print(f"Image : {w} x {h}")

    cv2.line(img, (BOTTOM_LINE_1, h), (TOP_LINE_1, 0), color=(0, 0, 255), thickness=2)
    cv2.line(img, (BOTTOM_LINE_2, h), (TOP_LINE_2, 0), color=(0, 0, 255), thickness=2)
    cv2.line(img, (BOTTOM_LINE_3, h), (TOP_LINE_3, 0), color=(0, 0, 255), thickness=2)
    cv2.line(img, (BOTTOM_LINE_4, h), (TOP_LINE_4, 0), color=(0, 0, 255), thickness=2)

    # Find intersection
    LAMBDA = (TOP_LINE_4 - TOP_LINE_1) / (BOTTOM_LINE_1 - TOP_LINE_1 - BOTTOM_LINE_4 + TOP_LINE_4)
    intersection = np.array([TOP_LINE_1 + LAMBDA * (BOTTOM_LINE_1 - TOP_LINE_1), LAMBDA * h])

    print("Intersection point:")
    print(intersection)

    # Here is the instrinc camera matrix
    f = 770.19284237
    cx = img.shape[1] / 2 - 0.5
    cy = img.shape[0] / 2 - 0.5
    K = np.zeros((3, 3))
    K[0, 0] = f
    K[1, 1] = f
    K[2, 2] = 1
    K[0, 2] = cx
    K[1, 2] = cy

    # For informational purpose
    h_fov = 2 * np.arctan(1280 / (2 * f)) * 180 / np.pi
    v_fov = 2 * np.arctan(720 / (2 * f)) * 180 / np.pi
    print(f"Horizontal fov = {h_fov:.2f} deg. Vertical fov = {v_fov:.2f} deg")

    intersection[1] = h - intersection[1]

    # Pitch and yaw estimates (see README.md)
    pitch_estimate = np.arctan((intersection[1] - cy) / f)
    print(f"Pitch estimate = {pitch_estimate* 180 / np.pi:.2f} deg")

    yaw_estimate = (
        (intersection[0] - img.shape[1] / 2) * np.cos(pitch_estimate) / K[0, 0]
    )
    print(f"Yaw estimate = {yaw_estimate* 180 / np.pi:.2f} deg")

    # Translation estimation (see README.md)
    ls = find_translation_from_points(
        K,
        np.array([BOTTOM_LINE_1, 0]),
        np.array([BOTTOM_LINE_2, 0]),
        np.array([BOTTOM_LINE_3, 0]),
        x_positions=[LINE_1_X, LINE_2_X, LINE_3_X],
    )

    # Refind original matrix
    est_to_camera_from_world = np.identity(4)
    pitch_matrix_estimate = make_euler(pitch_estimate, 0)
    yaw_matrix_estimate = make_euler(yaw_estimate, 1)
    est_to_camera_from_world[:3, :3] = pitch_matrix_estimate.dot(yaw_matrix_estimate)
    est_to_camera_from_world[:3, 3] = ls

    np.set_printoptions(suppress=True)
    print("Final extrinsic estimate:")
    to_world_from_camera = np.linalg.inv(est_to_camera_from_world)
    print(to_world_from_camera)

    # Reproject to check results

    line = Line(LINE_1_X)
    projection_1, projection_far_1 = line.project(to_world_from_camera, K)
    draw_line(img, projection_1, projection_far_1)

    line = Line(LINE_2_X)
    projection_2, projection_far_2 = line.project(to_world_from_camera, K)
    draw_line(img, projection_2, projection_far_2)

    line = Line(LINE_3_X)
    projection_3, projection_far_3 = line.project(to_world_from_camera, K)
    draw_line(img, projection_3, projection_far_3)

    line = Line(LINE_4_X)
    projection_4, projection_far_4 = line.project(to_world_from_camera, K)
    draw_line(img, projection_4, projection_far_4)

    line = Line(-0.95 - 3 * 3.45)
    projection_5, projection_far_5 = line.project(to_world_from_camera, K)
    draw_line(img, projection_5, projection_far_5)

    line = Line(0.95 + 3 * 3.45)
    projection_6, projection_far_6 = line.project(to_world_from_camera, K)
    draw_line(img, projection_6, projection_far_6)

    # Little loop to check how the bounding boxes perspectives
    img_with_lines = img.copy()

    while True:
        cv2.imshow("Image with car", img_with_lines)
        cv2.setMouseCallback(
            "Image with car",
            raycast,
            [img, K, to_world_from_camera],
        )

        k = cv2.waitKey(0)

        if k == 27:
            break
