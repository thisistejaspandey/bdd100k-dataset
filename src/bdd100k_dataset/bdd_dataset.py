from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
from loguru import logger

try:
    import orjson as json
except ImportError:
    import json


@dataclass
class BDD100kLabel:
    image_dir: Path
    numpy_cache_dir: Path
    remove_background: bool
    include_image_data: bool

    def format_labels_(self, label_data: dict):
        """
        - Converts list of labels of objects to a dictionary with the object id as the key.
        - Converts box2d to [x1, y1, x2, y2] format as numpy array.
        - Removes attributes.

        - Adds image data to the label data.

        Formats labels in-place.

        Returns:
            None
        """

        labels = label_data["labels"]

        formatted_labels = {}
        for label in labels:
            # Remove if object category is not in classes, thus background.
            category = label["category"]
            if self.remove_background and category not in self.classes:
                continue

            # Convert list of labels to dictionary with id as key.
            # Helps with creating object trajectories.
            id = label.pop("id")
            formatted_labels[id] = label

            # Convert box2d to [x1, y1, x2, y2]
            box2d = label["box2d"]
            formatted_labels[id]["box2d"] = np.asarray(
                [box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]]
            )

            # Remove attributes
            formatted_labels[id].pop("attributes", None)

        label_data["labels"] = formatted_labels

        # Add image data.
        if self.include_image_data:
            image_name = label_data["name"]
            video_name = label_data["videoName"]

            if self.numpy_cache_dir is not None:
                logger.warning(
                    "Numpy cache directory is not None. This is not implemented yet."
                )

            image_path = self.image_dir / video_name / image_name
            label_data["image_data"] = image_path.read_bytes()

        return label_data

    @property
    def classes(self):
        return {
            "pedestrian": 0,
            "rider": 1,
            "car": 2,
            "truck": 3,
            "bus": 4,
            "train": 5,
            "motorcycle": 6,
            "bicycle": 7,
        }


@dataclass
class BDD100KSequencer:
    window_size: int
    overlap: bool
    clean_background: bool
    clean_first: bool
    clean_trajectories: bool

    def create_windows(self, label_data: Iterable[dict]):
        """
        Creates windows of size `window_size` from the label data.

        """
        # Create a window of size `window_size`.
        window = deque(maxlen=self.window_size)

        # Store labels in the window.
        for label_data in label_data:
            # If a frame is empty, delete the entire trajectory.
            if self.clean_background and not len(label_data["labels"]):
                window.clear()
                continue

            # Add label data to the window.
            window.append(label_data)

            # If the window is full, validate the window.
            if len(window) == self.window_size:
                labels = self.validate_window(window)
                if labels is not None:
                    yield self.process_window(window, labels)

                if self.overlap:
                    window.popleft()
                else:
                    window.clear()

    def validate_window(self, window):
        objects = {}

        for frame_idx, frame_data in enumerate(window):
            for label_id, label in frame_data["labels"].items():
                objects.setdefault(label_id, {}).update({frame_idx: label})

        # Ensure object is present in the first frame.
        if self.clean_first:
            for object_id in list(objects.keys()):
                if 0 not in objects[object_id]:
                    objects.pop(object_id)

        # Check if the object is present in all frames.
        elif self.clean_trajectories:
            for object_id in list(objects.keys()):
                if len(objects[object_id]) != self.window_size:
                    objects.pop(object_id)

        # Check if any objects are present in the window.
        if not len(objects) and self.clean_background:
            return None

        return objects

    def process_window(self, window, labels):
        formatted_labels = {}
        for frame_idx, frame_data in enumerate(window):
            formatted_labels.setdefault("name", []).append(frame_data["name"])
            formatted_labels.setdefault("videoName", []).append(frame_data["videoName"])
            formatted_labels.setdefault("frameIndex", []).append(
                frame_data["frameIndex"]
            )

            if "image_data" in frame_data:
                formatted_labels.setdefault("image_data", []).append(
                    frame_data["image_data"]
                )

        formatted_labels["labels"] = labels

        return formatted_labels


@dataclass
class BDD100kVideoReader:
    bdd100k_root_dir: str | Path
    numpy_cache_dir: str | Path = None
    train: bool = False
    window_size: int = 1
    overlapping: bool = True
    remove_background: bool = False
    clean_first: bool = False
    clean_trajectories: bool = False
    include_image_data: bool = True

    def __post_init__(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")
        if self.numpy_cache is not None and not self.numpy_cache.exists():
            raise FileNotFoundError(
                f"Cache directory does not exist: {self.numpy_cache}"
            )

        if self.clean_first and self.clean_trajectories:
            raise ValueError("Both clean_first and clean_trajectories cannot be True.")

        if (self.clean_first or self.clean_trajectories) and not self.remove_background:
            warnings.warn(
                "clean_first or clean_trajectories is set to True but remove_background is set to False. Setting remove_background to True."
            )
            self.remove_background = True

    @property
    def classes(self):
        return {
            "pedestrian": 0,
            "rider": 1,
            "car": 2,
            "truck": 3,
            "bus": 4,
            "train": 5,
            "motorcycle": 6,
            "bicycle": 7,
        }

    @property
    def image_dir(self) -> Path:
        return (
            Path(self.bdd100k_root_dir)
            / "images/track"
            / ("train" if self.train else "val")
        )

    @property
    def label_dir(self) -> Path:
        return (
            Path(self.bdd100k_root_dir)
            / "labels/box_track_20"
            / ("train" if self.train else "val")
        )

    @property
    def numpy_cache(self) -> Path:
        if self.numpy_cache_dir is None:
            return None
        return Path(self.numpy_cache_dir) / "train.npy" if self.train else "val.npy"

    @property
    def train_length(self) -> int:
        return 278_079

    @property
    def val_length(self) -> int:
        return 39_973

    def __iter__(self):
        yield from self.process_labels()

    def __next__(self):
        raise RuntimeError("Create __iter__ method to iterate over the dataset.")

    def _create_window_splits_from_video_json(self, label_file: Path):
        # Load label data for an entire video sequence.
        json_data = json.loads(label_file.read_text())

        # Format label data, including adding image data.
        formatted_label_data = (
            BDD100kLabel(
                image_dir=self.image_dir,
                numpy_cache_dir=self.numpy_cache,
                remove_background=self.remove_background,
                include_image_data=self.include_image_data,
            ).format_labels_(label_data)
            for label_data in json_data
        )

        # Create window splits.
        yield from BDD100KSequencer(
            window_size=self.window_size,
            overlap=self.overlapping,
            clean_background=self.remove_background,
            clean_trajectories=self.clean_trajectories,
            clean_first=self.clean_first,
        ).create_windows(formatted_label_data)

    def process_labels(self):
        for label_file in self.label_dir.glob("*.json"):
            yield from self._create_window_splits_from_video_json(label_file)
