from pathlib import Path
from random import shuffle
import numpy as np
import xxhash

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIRaggedIterator

from bdd100k_dataset import BDD100kVideoReader


class BDD100kDaliReader:
    def __init__(
        self,
        bdd100k_root_dir: str | Path,
        numpy_cache_dir: str | Path,
        train: bool,
        window_size: int,
        overlapping: bool,
        remove_background: bool,
        clean_first: bool,
        clean_trajectories: bool,
        batch_size: int,
        device_id: int = 0,
        num_gpus: int = 1,
    ):
        self._dataset = BDD100kVideoReader(
            bdd100k_root_dir=bdd100k_root_dir,
            numpy_cache_dir=numpy_cache_dir,
            train=train,
            window_size=window_size,
            overlapping=overlapping,
            remove_background=remove_background,
            clean_first=clean_first,
            clean_trajectories=clean_trajectories,
            include_image_data=True,
        )

        # Shard the dataset.
        self.sequences = list(self._dataset)
        self.dataset_len = len(self.sequences)

        self.sequences = self.sequences[
            self.dataset_len * device_id // num_gpus : self.dataset_len
            * (device_id + 1)
            // num_gpus
        ]

        self.n = len(self.sequences)
        self.train = train
        self.batch_size = batch_size
        self.window_size = window_size

        # Drop the last batch if it's smaller than the batch size.
        last_batch_size = self.n % self.batch_size
        if last_batch_size:
            self.n -= last_batch_size
            self.sequences = self.sequences[:-last_batch_size]

        if self.train:
            self.shuffle()

    @property
    def num_iterations(self):
        return self.n // self.batch_size

    def __iter__(self):
        return (
            self.sequences[i : i + self.batch_size]
            for i in range(0, self.n, self.batch_size)
        )

    def get_imagedata_and_metadata_readers(self):
        return ImageDataIterator(self), MetadataIterator(self)

    def shuffle(self):
        shuffle(self.sequences)


class BaseIterator:
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.curr_iterator = self.reset()

    def hash_image_buffer(self, image_buffer_list):
        return np.asarray(
            [
                np.asarray([xxhash.xxh32_intdigest(image_buffer)])
                for image_buffer in image_buffer_list
            ],
            dtype=np.int32,
        )

    def reset(self):
        return iter(self.base_dataset)

    def __iter__(self):
        self.curr_iterator = self.reset()
        return self

    def __next__(self):
        raise NotImplementedError


class ImageDataIterator(BaseIterator):
    def __next__(self):
        data = next(self.curr_iterator)
        flattened_image_data = [img_data for d in data for img_data in d["image_data"]]
        image_hashes = self.hash_image_buffer(flattened_image_data)
        return (flattened_image_data, image_hashes)


class MetadataIterator(BaseIterator):
    def __next__(self):
        data = next(self.curr_iterator)
        labels = [d["labels"] for d in data]
        full_names = [
            [
                f"{video_name}/{name}"
                for name, video_name in zip(d["name"], d["videoName"])
            ]
            for d in data
        ]

        flattened_image_data = [img_data for d in data for img_data in d["image_data"]]
        image_hashes = self.hash_image_buffer(flattened_image_data)

        return {"filenames": full_names, "labels": labels, "image_hashes": image_hashes}


class BDD100kDaliDataset:
    def __init__(
        self,
        bdd100k_root_dir: str | Path,
        numpy_cache_dir: str | Path = None,
        train: bool = False,
        window_size: int = 1,
        overlapping: bool = True,
        remove_background: bool = False,
        clean_first: bool = False,
        clean_trajectories: bool = False,
        batch_size: int = 1,
        local_rank: int = 0,
        global_rank: int = 0,
        num_gpus: int = 1,
        num_threads: int = 8,
    ):
        self.bdd_reader = BDD100kDaliReader(
            bdd100k_root_dir=bdd100k_root_dir,
            numpy_cache_dir=numpy_cache_dir,
            train=train,
            window_size=window_size,
            overlapping=overlapping,
            remove_background=remove_background,
            clean_first=clean_first,
            clean_trajectories=clean_trajectories,
            batch_size=batch_size,
            device_id=global_rank,
            num_gpus=num_gpus,
        )

        self.imagedata_reader, self.metadata_reader = (
            self.bdd_reader.get_imagedata_and_metadata_readers()
        )

        self.dali_pipeline = self.pipe(
            batch_size=batch_size * window_size,
            num_threads=num_threads,
            device_id=local_rank,
        )

        outputs = ["images", "image_hashes"]

        self.pytorch_iterator = BDD100kPytorchIterator(
            self.dali_pipeline,
            output_map=outputs,
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
                DALIRaggedIterator.DENSE_TAG,
            ],
            window_size=window_size,
            metadata_reader=self.metadata_reader,
            bdd_reader=self.bdd_reader,
        )

    @pipeline_def
    def pipe(self):
        jpegs, hashes = fn.external_source(
            source=self.imagedata_reader,
            num_outputs=2,
            cycle="raise",
        )
        images = fn.decoders.image(jpegs, device="mixed")
        return images, hashes

    def __iter__(self):
        return self.pytorch_iterator.__iter__()

    def __len__(self):
        return self.reader.num_iterations


class BDD100kPytorchIterator(DALIRaggedIterator):
    def __init__(self, *args, **kwargs):
        self.window_size = kwargs.pop("window_size")
        self.metadata_reader = kwargs.pop("metadata_reader")
        self.bdd_reader = kwargs.pop("bdd_reader")
        super().__init__(*args, **kwargs)

    def __next__(self):
        try:
            data = super().__next__()
            metadata = self.metadata_reader.__next__()
        except StopIteration:
            if self.bdd_reader.train:
                self.bdd_reader.shuffle()
            self.metadata_reader = iter(self.metadata_reader)
            raise StopIteration

        filenames = metadata["filenames"]
        labels = metadata["labels"]
        hashes = metadata["image_hashes"]

        images = data[0]["images"]
        images = images.unflatten(0, (-1, self.window_size))
        image_hashes = data[0]["image_hashes"]

        return {
            "images": images,
            "filenames": filenames,
            "labels": labels,
            "dali_hashes": image_hashes,
            "meta_hashes": hashes,
        }
