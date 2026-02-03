"""Video clip extraction and augmentation utilities."""

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

_WARNED_FILES = set()


def _warn_once(msg):
    if msg in _WARNED_FILES:
        return
    _WARNED_FILES.add(msg)
    print(msg)

# https://discuss.pytorch.org/t/speed-up-dataloader-using-the-new-torchvision-transforms-support-for-tensor-batch-computation-gpu/113166
Train_Transforms = torch.nn.Sequential(
    v2.RandomZoomOut(fill=0, side_range=(1.0, 1.2), p=0.5),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(-15, 15)),
    v2.CenterCrop((224, 224)),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet norm RGB
)

Debug_Transforms_noNorm = torch.nn.Sequential(  # for visualizing
    v2.RandomZoomOut(fill=0, side_range=(1.0, 1.2), p=0.5),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(-15, 15)),
    v2.CenterCrop((224, 224)),
    # v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet norm RGB
)

Test_Transforms = torch.nn.Sequential(
    v2.CenterCrop((224, 224)),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet norm RGB
)


def pull_clips(file_loc, transform_func, num_clips=16, clip_len=16):
    """Sample multiple random clips from a video and stack them.

    Args:
        file_loc (str): Path to a video file.
        transform_func (callable): Transform pipeline for clips.
        num_clips (int): Number of clips to sample.
        clip_len (int): Frames per clip.

    Returns:
        torch.Tensor: Stacked clips with shape (num_clips, 3, clip_len, 224, 224).
    """
    clips = [pull_clip(file_loc, transform_func, clip_len=clip_len) for _ in range(num_clips)]
    return torch.vstack(clips)


def pull_clip(file_loc, transform_func, clip_len=16):
    """Sample a random clip and apply transforms.

    Args:
        file_loc (str): Path to a video file.
        transform_func (callable): Transform pipeline for clips.
        clip_len (int): Frames per clip.

    Returns:
        torch.Tensor: Clip tensor shaped (1, 3, clip_len, 224, 224).
    """
    capture = cv2.VideoCapture(file_loc)
    if not capture.isOpened():
        _warn_once(f"warning: failed to open video {file_loc}")
        frames = np.zeros((clip_len, 256, 256, 3), dtype=np.uint8)
        frames = tv_tensors.Video(np.transpose(frames, (0, 3, 1, 2)))
        frames = frames.type(torch.float32) / 255
        frames = transform_func(frames)
        clip = frames.unsqueeze(0).transpose(1, 2)
        return clip
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < clip_len:
        start_idx = 0
    else:
        start_idx = np.random.randint(0, frame_count - clip_len + 1, size=1)[0]

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx - 1)

    frames = []
    last_frame = None
    for i in range(clip_len):
        if i < frame_count:
            ret, frame = capture.read()
            if not ret or frame is None:
                _warn_once(f"warning: failed to read frame from {file_loc}")
                if last_frame is None:
                    # Fallback to a black frame if the first read fails
                    frame = np.zeros((256, 256, 3), dtype=np.uint8)
                else:
                    frame = last_frame
            else:
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            last_frame = frame
            frames.append(frame)
        else:
            # "last image carried forward"
            if last_frame is None:
                last_frame = np.zeros((256, 256, 3), dtype=np.uint8)
            frames.append(last_frame)

    frames = np.stack(frames, axis=0)  # f x h x w x 3
    frames = tv_tensors.Video(np.transpose(frames, (0, 3, 1, 2)))  # f x 3 x h x w

    frames = frames.type(torch.float32) / 255  # convert to 0-1 float32 format
    frames = transform_func(frames)

    clip = frames.unsqueeze(0).transpose(1, 2)

    return clip
