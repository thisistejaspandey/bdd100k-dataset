from pathlib import Path
from random import shuffle

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

    def __iter__(self):
        if self.train:
            shuffle(self.sequences)

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n - self.batch_size + 1:
            self.__iter__()
            raise StopIteration

        batched_sequences = self.sequences[self.i : self.i + self.batch_size]
        batched_sequences_frame_data = [
            sequence[i].pop("image_data")
            for sequence in batched_sequences
            for i in range(self.window_size)
        ]

        self.i += self.batch_size
        return batched_sequences_frame_data


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
        self.reader = BDD100kDaliReader(
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

        self.dali_pipeline = self.pipe(
            batch_size=batch_size * window_size,
            num_threads=num_threads,
            device_id=local_rank,
        )

        outputs = ["images"]

        self.pytorch_iterator = BDD100kPytorchIterator(
            self.dali_pipeline,
            outputs,
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
                # DALIRaggedIterator.SPARSE_LIST_TAG,
            ],
        )

    @pipeline_def
    def pipe(self):
        jpegs = fn.external_source(
            source=self.reader,
            num_outputs=1,
            cycle="raise",
        )

        images = fn.decoders.image(jpegs, device="mixed")

        return images


class BDD100kPytorchIterator(DALIRaggedIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = 1

    def __next__(self):
        data = super().__next__()

        images = data[0]["images"]

        images = images.unflatten(0, (-1, self.window_size))

        return images
