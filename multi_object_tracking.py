"""
Perform multi object tracking on a mp4 video with a pytorch neural network model (.pth)
Instead of only performing detections, we try to determine objects position over time.
"""

from argparse import ArgumentParser
from time import time
import numpy as np
import cv2
from tqdm import tqdm

from utils import draw_fps, draw_frame_id, COLORS, Box, get_position_ground, make_euler, project
from object_detector import ObjectDetector
from kalman_filter import KalmanFilter

to_world_from_camera = np.identity(4)
pitch_matrix = make_euler(-27.39 * np.pi / 180, 0)
to_world_from_camera[:3, :3] = pitch_matrix
to_world_from_camera[:3, 3] = np.array([0, 8.5, 0])

# (from CARLA)
K = np.array([[770.19284237, 0.0, 640.0], [0.0, 770.19284237, 360.0], [0.0, 0.0, 1.0]])


class Object:
    """
    Object class with its current state (position, yaw, width/height),
    age since creation, trajectory, Kalman Filter.
    We also have an unassociated counter to be robust to misdetections
    """

    def __init__(self, id, det):
        self.id = id
        self.age = 0
        self.unassociated_counter = 0
        self.kf = KalmanFilter(
            nx=6,
            nz=5,
            first_measurement=np.asarray(
                [det["x"], det["y"], det["w"], det["h"], det["l"]]
            ).reshape(-1),
        )

    def dist(self, det):
        """Compute distance between object center and detection"""
        return np.linalg.norm(np.array([self.x() - det["x"], self.y() - det["y"]]))

    def update(self, det):
        """Update object after a match with a detection"""

        meas = np.asarray([det["x"], det["y"], det["w"], det["h"], det["l"]])

        self.kf.update(meas)
        self.unassociated_counter = 0

    def x(self):
        return self.kf.estimate[0, 0]

    def y(self):
        return self.kf.estimate[1, 0]

    def w(self):
        return self.kf.estimate[3, 0]

    def h(self):
        return self.kf.estimate[4, 0]

    def l(self):
        return self.kf.estimate[5, 0]

    def get_speed(self):
        return self.kf.estimate[2, 0]

    def get_position(self):
        return self.kf.estimate[0:2, 0]


class Tracking:
    """
    Tracking class: responsible to create/kill objects,
    and to match current detections with previous Tracking state
    """

    def __init__(self):
        self.objects = []
        self.last_id = -1

    def add(self, det):
        """Add a new object"""
        obj = Object(self.last_id + 1, det)
        self.objects.append(obj)
        self.last_id += 1

    def kill(self, id):
        """Kill object with id"""
        next_objects = []
        for object in self.objects:
            if object.id != id:
                next_objects.append(object)

        self.objects = next_objects

    def print_match(self, detections):
        """Debug only"""
        print("Match :")
        matches = np.zeros((len(self.objects), len(detections)))
        for i, object in enumerate(self.objects):
            for j, detection in enumerate(detections):
                matches[i, j] = object.dist(detection)

        print(matches)

    def hungarian(self, detections):
        """Compute distances between detection bboxes and tracking objects"""
        distances = np.zeros((len(self.objects), len(detections)))
        for i, object in enumerate(self.objects):
            for j, detection in enumerate(detections):
                distances[i, j] = object.dist(detection)

        return distances

    def update(self, detections):
        """
        Update tracking with latest detections (from Pytorch model)

        """

        # First predict objects state with KF
        for object in self.objects:
            object.kf.predict()

        # Compute distances between detections and tracking objects
        impossible = 1000
        distances = self.hungarian(detections)
        if len(distances) == 0:
            # No objects yet
            for detection in detections:
                print(f"Create new object with det {detection}")
                self.add(detection)
            return

        # Match objects and detections if their respective distance is < 30px
        match_objects = [False] * len(self.objects)
        match_detections = [False] * len(detections)

        if distances.shape[1] > 0:
            while True:
                assert distances.shape[0] > 0
                assert distances.shape[1] > 0
                match_idx = np.asarray(
                    np.unravel_index(np.argmin(distances), distances.shape)
                )
                i = match_idx[0]
                j = match_idx[1]
                if distances[i, j] > 2:
                    break

                if abs(detections[j]["x"] - self.objects[i].x()) > 1:
                    break

                self.objects[i].update(detections[j])
                distances[i, :] = impossible
                distances[:, j] = impossible
                match_objects[i] = True
                match_detections[j] = True

        # Create objects if detection didnt match any previous object
        for j, detection in enumerate(detections):
            if not match_detections[j]:
                print(f"Create new object with det {detection}")
                self.add(detection)

        # Update age
        for object in self.objects:
            object.age += 1

        # Delete objects that have not matched with any detections for 20 scans
        objects = self.objects.copy()
        for i, match_object in enumerate(match_objects):
            if not match_object:
                # Kill object
                position = objects[i].get_position()
                objects[i].unassociated_counter += 1
                # Delete objects that go out of bounds or that are very young
                if objects[i].age < 5 or position[0] < -11 or position[0] >= 11:
                    print(
                        f"Kill object {objects[i].id} with age {objects[i].age} at position "
                        f"{position} after {objects[i].unassociated_counter} unassociated frames"
                    )
                    self.kill(objects[i].id)

                elif objects[i].unassociated_counter >= 10:
                    print(
                        f"!!!!!! Kill object {objects[i].id} with age {objects[i].age} at position "
                        f"{position} after {objects[i].unassociated_counter} unassociated frames"
                    )
                    # assert False
                    self.kill(objects[i].id)

        for object in objects:
            if object.y() > 40:
                self.kill(object.id)

        # Kill conflicts
        objects = self.objects.copy()
        for i, obj_1 in enumerate(objects):
            for j in range(i + 1, len(objects)):
                obj_2 = objects[j]
                if (obj_1.x() - obj_2.x()) ** 2 + (obj_1.y() - obj_2.y()) ** 2 < 9:
                    print("Conflict", obj_1.id, obj_2.id)
                    if obj_1.age < obj_2.age:
                        self.kill(obj_1.id)
                    else:
                        self.kill(obj_2.id)

    def __str__(self) -> str:
        if len(self.objects) == 0:
            return "Tracking is empty"

        str = "Tracking contains:\n"
        str += "------------------------------------------------------\n"
        str += (
            "   Id   |   X   |   Y   |   W   |   H   |   L   |  Age  |  US  | Speed |\n"
        )
        str += "------------------------------------------------------\n"
        for object in self.objects:
            str += f"{object.id:5}   | {object.x():.1f}  | {object.y():.1f}  |"
            str += f" {object.w():.1f}  | {object.h():.1f}  |  {object.l():.1f}  |"
            str += f"{object.age}  |  {object.unassociated_counter}  |  {object.get_speed():.1f}\n"
        return str

    def display(self, frame):
        """
        Draw each objects with oriented bounding boxes and trajectory
        """
        to_camera_from_world = np.linalg.inv(to_world_from_camera)
        for object in self.objects:
            box = Box(
                object.x(),
                object.y(),
                object.w(),
                object.h(),
                object.l(),
            )
            color = COLORS[object.id % len(COLORS)]

            pix = project(
                np.array([object.x(), 0, object.y()]), to_camera_from_world, K
            )

            frame = box.project(frame, to_world_from_camera, K, color=color)

            # cv2.putText(
            #    frame,
            #    f"{object.id}",
            #    (int(pix[0]), 720 - int(pix[1])),
            #    cv2.FONT_HERSHEY_SIMPLEX,
            #    fontScale=0.8,
            #    color=color,
            #    thickness=2,
            # )

            cv2.putText(
                frame,
                f"{abs(object.get_speed()*3.6):.0f} km/h",
                (int(pix[0]) - 40, 720 - int(pix[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=2,
            )

        return frame


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object tracking")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument("model", type=str, help="Pytorch model for bbox cars detection")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Threshold to keep an object"
    )
    parser.add_argument(
        "-f", type=int, default=np.inf, help="Pause viz at specified frame"
    )
    args = parser.parse_args()

    object_detector = ObjectDetector(args.model, args.conf)

    cap = cv2.VideoCapture(args.video)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video stream or file")

    prev_time = time()

    tracking = Tracking()
    PLAY = False

    for idx in tqdm(range(n_frames)):
        if not cap.isOpened():
            break

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n#################### Frame {idx} #####################")

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bboxs = object_detector.detect(frame)
        # frame = showbox(frame, bboxs)

        world_bboxs = []

        for box in bboxs:
            position_on_ground = get_position_ground(
                box["x"], box["y"], K, to_world_from_camera, 720
            )

            if abs(position_on_ground[0]) > 11:
                continue

            depth = np.linalg.norm(position_on_ground - to_world_from_camera[:3, 3])

            if box["w"] * depth < 1:
                continue

            if box["h"] * depth < 1 or box["h"] * depth > 5:
                continue

            if box["l"] * depth < 3:
                continue

            if box["h"] > box["l"]:
                continue

            world_bbox = {
                "x": position_on_ground[0],
                "y": position_on_ground[2],
                "w": min(box["w"] * depth, 2.5),
                "h": box["h"] * depth,
                "l": box["l"] * depth,
            }

            world_bboxs.append(world_bbox)

        tracking.update(world_bboxs)
        print(tracking)

        # frame = showbox(frame, bboxs)
        frame = tracking.display(frame)

        fps = 1 / (time() - prev_time)
        prev_time = time()
        draw_fps(frame, fps)
        draw_frame_id(frame, idx)

        cv2.imshow("Detection", frame)
        k = cv2.waitKey(10 if ((idx < args.f < np.inf) or PLAY) else 0)

        if k == 27:
            break

        if k == 32:  # SPACE
            PLAY = not PLAY

        # cv2.imwrite(f"video_tracking_carla/image_{idx:04d}.png", frame)
