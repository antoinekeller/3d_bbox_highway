"""
Implement object detector class with helper functions
Load model, preprocess image, infer, post-process results
And returns detected bboxs.
"""
from argparse import ArgumentParser
import cv2
import numpy as np

from utils import Box, get_position_ground, make_euler
from object_detector import ObjectDetector


to_world_from_camera = np.identity(4)
pitch_matrix = make_euler(-27.39 * np.pi / 180, 0)
to_world_from_camera[:3, :3] = pitch_matrix
to_world_from_camera[:3, 3] = np.array([0, 8.5, 0])

# (from CARLA)
K = np.array([[770.19284237, 0.0, 640.0], [0.0, 770.19284237, 360.0], [0.0, 0.0, 1.0]])


def showbox(img, boxes):
    """
    Convert bounding boxes from pixel coordinates and dimensions that is normalized by depth
    to world bounding box
    """
    for box in boxes:
        # First find position in world
        position_on_ground = get_position_ground(
            box["x"], box["y"], K, to_world_from_camera, img.shape[0]
        )

        # Compute depth at this point (distance from camera)
        depth = np.linalg.norm(position_on_ground - to_world_from_camera[:3, 3])

        # Discard unrealistic bbox
        if box["w"] * depth < 1:
            continue

        if box["h"] * depth < 1:
            continue

        if box["l"] * depth < 1:
            continue

        # Multiply width/height/length by depth
        box = Box(
            position_on_ground[0],
            position_on_ground[2],
            box["w"] * depth,
            box["h"] * depth,
            box["l"] * depth,
        )
        # print(label)
        # Re-project them
        img = box.project(img, to_world_from_camera, K)

    return img


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object detection")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument(
        "model", type=str, help="Pytorch model for oriented cars bbox detection"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Threshold to keep an object"
    )
    args = parser.parse_args()

    object_detector = ObjectDetector(args.model, args.conf)

    cap = cv2.VideoCapture(args.video)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxs = object_detector.detect(frame)
        frame = showbox(frame, bboxs)

        cv2.imshow("Detection", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
