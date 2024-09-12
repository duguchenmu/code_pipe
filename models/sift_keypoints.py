#import pycolmap
import torch
import numpy as np
import cv2
# def run_opencv_sift(features: cv2.Feature2D, image: np.ndarray) -> np.ndarray:
#     """
#     Detect keypoints using OpenCV Detector.
#     Optionally, perform description.
#     Args:
#         features: OpenCV based keypoints detector and descriptor
#         image: Grayscale image of uint8 data type
#     Returns:
#         keypoints: 1D array of detected cv2.KeyPoint
#         scores: 1D array of responses
#         descriptors: 1D array of descriptors
#     """
#     detections, descriptors = features.detectAndCompute(image, None)
#     points = np.array([k.pt for k in detections], dtype=np.float32)
#     scores = np.array([k.response for k in detections], dtype=np.float32)
#     scales = np.array([k.size for k in detections], dtype=np.float32)
#     angles = np.deg2rad(np.array([k.angle for k in detections], dtype=np.float32))
#     return points, scores, scales, angles, descriptors
def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
    x = torch.transpose(x, 1, 2)
    return x
def extract_single_image(x):
    sift = cv2.SIFT_create()
    x = (x.squeeze().squeeze().numpy() * 255.0).astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    # # keypoints, scores, scales, angles, descriptors = run_opencv_sift(
    #     sift, x
    # )
    detections, descriptors = sift.detectAndCompute(x, None)
    points = np.array([k.pt for k in detections], dtype=np.float32)
    scores = np.array([k.response for k in detections], dtype=np.float32)
    scales = np.array([k.size for k in detections], dtype=np.float32)
    angles = np.deg2rad(np.array([k.angle for k in detections], dtype=np.float32))

    pred = {
        "keypoints": points,
        "scales": scales,
        "oris": angles,
        "descriptors": descriptors,
        "keypoint_scores": scores,
    }

    pred = {k: torch.from_numpy(v) for k, v in pred.items()}

    if scores is not None:
        # Keep the k keypoints with highest score
        num_points = 4096
        if num_points is not None and len(pred["keypoints"]) > num_points:
            indices = torch.topk(pred["keypoint_scores"], num_points).indices
            pred = {k: v[indices] for k, v in pred.items()}
    return pred
def detect_sift_keypoint(x):

        # sift = pycolmap.Sift()
        # detections, descriptors = sift.extract(x['image'])
        # keypoints = detections[:, :2]
        # scales, angles = detections[:, -2:].T
        # pred = {
        #     "keypoints": keypoints,
        #     "scales": scales,
        #     "oris": angles,
        #     "descriptors": descriptors,
        # }
        # # 移除外边关键点
        # is_inside = (pred["keypoints"] + 0.5 < np.array([x.shape[-2:][::-1]])).all(-1)
        # pred = {k: v[is_inside] for k, v in pred.items()}
        #
        # pred = {k: torch.from_numpy(v) for k, v in pred.items()}
        # return pred
        pred = []
        p = extract_single_image(x)
        pred.append(p)
        pred = {k: torch.stack([p[k] for p in pred], 0) for k in pred[0]}
        pred["descriptors"] = sift_to_rootsift(pred["descriptors"])
        return pred