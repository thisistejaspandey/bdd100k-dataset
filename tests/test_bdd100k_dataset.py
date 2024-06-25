import pytest
from pathlib import Path
from bdd100k_dataset.bdd_dataset import (
    BDD100kLabel,
    BDD100kVideoReader,
)
from icecream import ic

import numpy as np


def compare_dicts(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1:
        val1, val2 = dict1[key], dict2[key]

        if type(val1) != type(val2):
            return False
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dicts(val1, val2):
                return False
        elif val1 != val2:
            return False
    return True


@pytest.fixture
def mock_data():
    mock_label_data = {
        "name": "b1c66a42-6f7d68ca-0000001.jpg",
        "labels": [
            {
                "id": "00122062",
                "category": "car",
                "attributes": {"occluded": False, "truncated": True, "crowd": False},
                "box2d": {
                    "x1": 0,
                    "x2": 75.41276162779963,
                    "y1": 346.55482900742646,
                    "y2": 407.4496307987563,
                },
            },
            {
                "id": "00122063",
                "category": "car",
                "attributes": {"occluded": True, "truncated": False, "crowd": False},
                "box2d": {
                    "x1": 63.71773346919986,
                    "x2": 130.258410923302,
                    "y1": 350.58759733797814,
                    "y2": 402.2070319690392,
                },
            },
            {
                "id": "00122064",
                "category": "boat",
                "attributes": {"occluded": True, "truncated": False, "crowd": False},
                "box2d": {
                    "x1": 85.09140562112357,
                    "x2": 222.60880569293465,
                    "y1": 354.62036566852976,
                    "y2": 419.54793579041126,
                },
            },
        ],
        "videoName": "b1c66a42-6f7d68ca",
        "frameIndex": 0,
    }

    return mock_label_data


def test_format_labels_(mock_data):
    expected_formatted_labels = {"1": {"category": "car", "box2d": [0, 0, 10, 10]}}
    expected_formatted_labels = {
        "name": "b1c66a42-6f7d68ca-0000001.jpg",
        "labels": {
            "00122062": {
                "category": "car",
                "box2d": np.array(
                    [0, 346.55482900742646, 75.41276162779963, 407.4496307987563]
                ),
            },
            "00122063": {
                "category": "car",
                "box2d": np.array(
                    [
                        63.71773346919986,
                        350.58759733797814,
                        130.258410923302,
                        402.2070319690392,
                    ]
                ),
            },
            "00122064": {
                "category": "boat",
                "box2d": np.array(
                    [
                        85.09140562112357,
                        354.62036566852976,
                        222.60880569293465,
                        419.54793579041126,
                    ]
                ),
            },
        },
        "videoName": "b1c66a42-6f7d68ca",
        "frameIndex": 0,
    }

    # mock_image_dir = Path("/mock/image/dir")

    mock_image_dir = Path("/mock/image/dir")
    mock_numpy_cache_dir = Path("/mock/numpy/cache/dir")
    mock_label = BDD100kLabel(
        image_dir=mock_image_dir,
        numpy_cache_dir=mock_numpy_cache_dir,
        remove_background=False,
        include_image_data=False,
    )

    assert compare_dicts(
        mock_label.format_labels_(mock_data), expected_formatted_labels
    )


def test_format_labels_remove_background(mock_data):
    expected_formatted_labels = {
        "name": "b1c66a42-6f7d68ca-0000001.jpg",
        "labels": {
            "00122062": {
                "category": "car",
                "box2d": np.array(
                    [0, 346.55482900742646, 75.41276162779963, 407.4496307987563]
                ),
            },
            "00122063": {
                "category": "car",
                "box2d": np.array(
                    [
                        63.71773346919986,
                        350.58759733797814,
                        130.258410923302,
                        402.2070319690392,
                    ]
                ),
            },
        },
        "videoName": "b1c66a42-6f7d68ca",
        "frameIndex": 0,
    }

    # mock_image_dir = Path("/mock/image/dir")

    mock_image_dir = Path("/mock/image/dir")
    mock_numpy_cache_dir = Path("/mock/numpy/cache/dir")
    mock_label = BDD100kLabel(
        image_dir=mock_image_dir,
        numpy_cache_dir=mock_numpy_cache_dir,
        remove_background=True,
        include_image_data=False,
    )

    assert compare_dicts(
        mock_label.format_labels_(mock_data), expected_formatted_labels
    )


def test_bdd100k_dataset_remove_background():
    classes = {
        "pedestrian": 0,
        "rider": 1,
        "car": 2,
        "truck": 3,
        "bus": 4,
        "train": 5,
        "motorcycle": 6,
        "bicycle": 7,
    }

    # Test remove background.
    dataset = BDD100kVideoReader(
        "/home/tejas/gits/datasets/bdd100k/",
        window_size=5,
        train=False,
        include_image_data=False,
        remove_background=True,
    )

    for data in dataset:
        for label_data in data['labels'].values():
            for frame_data in label_data.values():
                assert frame_data['category'] in classes
    

def test_bdd100k_dataset_clean_trajectories():
    # Test clean trajectories.
    window_size = 5
    dataset = BDD100kVideoReader(
        "/home/tejas/gits/datasets/bdd100k/",
        window_size=window_size,
        train=False,
        include_image_data=False,
        clean_trajectories=True,
        remove_background=True,
    )

    for data in dataset:
        for label_data in data['labels'].values():
            assert len(label_data.keys()) == window_size


def test_bdd100k_dataset_clean_first():
    # Test clean trajectories.
    window_size = 5
    dataset = BDD100kVideoReader(
        "/home/tejas/gits/datasets/bdd100k/",
        window_size=window_size,
        train=False,
        include_image_data=False,
        clean_first=True,
        remove_background=True,
    )

    for data in dataset:
        for label_data in data['labels'].values():
            assert 0 in label_data.keys()







            

