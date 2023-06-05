import torch
from torch import nn
import torchvision

from torchvision import transforms
import numpy as np


def select(hm, threshold):
    """
    Keep only local maxima (kind of NMS).
    We make sure to have no adjacent detection in the heatmap.
    """

    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    for i, ci in enumerate(pred_centers):
        for j in range(i + 1, len(pred_centers)):
            cj = pred_centers[j]
            if np.linalg.norm(ci - cj) <= 2:
                score_i = hm[ci[0], ci[1]]
                score_j = hm[cj[0], cj[1]]
                if score_i > score_j:
                    hm[cj[0], cj[1]] = 0
                else:
                    hm[ci[0], ci[1]] = 0

    return hm


class ObjectDetector:
    """
    Initialize ObjectDetector by loading model
    Define confidence threshold
    Then preprocess, infer, post-process
    """

    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MODEL_SCALE = 32

    def __init__(self, model_pth, conf_threhsold):
        self.conf = conf_threhsold
        assert torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")

        # Define and load model
        self.model = centernet()
        self.model.load_state_dict(torch.load(model_pth))
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def detect(self, frame):
        """
        Detect objects on frame and return bounding boxes.
        First resize image to expected shape.
        Pre-process for model input
        Extract heatmaps, and compute predicted bounding boxes
        """

        input_tensor = self.preprocess(frame)

        # Inference
        hm, offset, whl = self.model(input_tensor.to(self.device).float().unsqueeze(0))

        hm = torch.sigmoid(hm)

        hm = hm.cpu().detach().numpy().squeeze(0).squeeze(0)
        offset = offset.cpu().detach().numpy().squeeze(0)
        whl = whl.cpu().detach().numpy().squeeze(0)

        hm = select(hm, self.conf)

        boxes = self.pred2box(hm, offset, whl, self.conf)

        return boxes

    def pred2box(self, hm, offset, regr, thresh=0.99):
        # make binding box from heatmaps
        # thresh: threshold for logits.

        # get center
        pred = hm > thresh
        pred_center = np.where(hm > thresh)

        # get regressions
        pred_r = regr[:, pred].T

        # wrap as boxes
        # [xmin, ymin, width, height]
        # size as original image.
        boxes = []

        pred_center = np.asarray(pred_center).T
        # print(pred_r.shape)
        # print(pred_angles)
        # print(pred_angles.shape)

        for center, b in zip(pred_center, pred_r):
            # print(b)
            offset_xy = offset[:, center[0], center[1]]
            bbox = {
                "x": (center[1] + offset_xy[0]) * self.MODEL_SCALE,
                "y": (center[0] + offset_xy[1]) * self.MODEL_SCALE,
                "w": b[0],
                "h": b[1],
                "l": b[2],
            }

            # discard negative values
            if bbox["w"] < 0:
                continue

            if bbox["h"] < 0:
                continue

            if bbox["l"] < 0:
                continue

            boxes.append(bbox)

        return boxes


class centernet(nn.Module):
    """
    Centernet simplified version
    Input = 1280x720 RGB image
    Output = 4 heatmaps
    * Main = [1, 45, 80]
    * Offset = [2, 45, 80]
    * Width/Height = [2, 45, 80]
    * Cos/sin angle = [2, 45, 80]
    """

    def __init__(self):
        super().__init__()

        # Resnet-18 as backbone.
        basemodel = torchvision.models.resnet18(weights=None)

        # Select only first layers up when you reach 160x90 dimensions with 256 channels
        self.base_model = nn.Sequential(*list(basemodel.children())[:-2])

        num_ch = 512
        head_conv = 64
        self.outc = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1),
        )

        self.outo = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1),
        )

        self.outr = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 3, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # [b, 3, 720, 1280]

        x = self.base_model(x)
        # [b, 128, 45, 80]

        assert not torch.isnan(x).any()

        outc = self.outc(x)
        # [b, 1, 45, 80]
        assert not torch.isnan(outc).any()

        outo = self.outo(x)
        # [b, 2, 45, 80]
        assert not torch.isnan(outo).any()

        outr = self.outr(x)

        return outc, outo, outr
