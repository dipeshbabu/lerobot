# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import weakref
from itertools import islice
from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader, Dataset, IterableDataset

pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from accelerate.data_loader import prepare_data_loader  # noqa: E402

from lerobot.datasets.sampler import EpisodeAwareSampler, compute_sampler_state  # noqa: E402
from lerobot.scripts import lerobot_train as training  # noqa: E402

pytestmark = pytest.mark.dataset


class _Frames(Dataset):
    def __init__(self, size):
        self.size = size
        self.reads = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.reads.append(index)
        return index


def _prepare(size, batch_size, world, rank, state=None):
    dataset = _Frames(size)
    sampler = EpisodeAwareSampler([0], [size], shuffle=True, seed=42)
    if state is not None:
        sampler.load_state_dict(state)
    loader = prepare_data_loader(
        DataLoader(dataset, batch_size=batch_size, sampler=sampler),
        num_processes=world,
        process_index=rank,
        rng_types=[],
    )
    return dataset, sampler, loader


@pytest.mark.parametrize("size,batch_size,world", [(10, 2, 1), (10, 2, 3), (11, 4, 2), (2, 4, 3)])
def test_prepared_resume_matches_uninterrupted_batches(size, batch_size, world):
    for rank in range(world):
        _, _, reference = _prepare(size, batch_size, world, rank)
        epochs = [[batch.tolist() for batch in reference] for _ in range(4)]
        batches_per_epoch = len(epochs[0])
        for step in range(3 * batches_per_epoch):
            epoch, offset = divmod(step, batches_per_epoch)
            state = compute_sampler_state(step, size, batch_size, world)
            dataset, sampler, resumed = _prepare(size, batch_size, world, rank, state)
            expected = epochs[epoch][offset:] + epochs[epoch + 1]
            actual = [
                batch.tolist()
                for batch in islice(training._cycle_training_dataloader(resumed, sampler), len(expected))
            ]
            assert actual == expected, (rank, step)
            # Skipping earlier batches must only walk indices, without loading samples.
            assert dataset.reads == [index for batch in expected for index in batch]


def test_prepared_resume_keeps_unaligned_offset_after_batch_size_change():
    _, sampler, loader = _prepare(11, 3, 1, 0, {"epoch": 2, "start_index": 4})
    reference = EpisodeAwareSampler([0], [11], shuffle=True, seed=42)
    reference.set_epoch(2)
    expected = list(reference)[4:]
    actual = list(islice(training._cycle_training_dataloader(loader, sampler), 3))
    assert [index for batch in actual for index in batch.tolist()] == expected


def test_training_dataloader_resume_matches_original_shard(monkeypatch, tmp_path):
    dataset = _Frames(10)
    dataset.meta = SimpleNamespace(
        episodes={"dataset_from_index": [0], "dataset_to_index": [10]}, has_language_columns=False
    )
    dataset.episodes = None
    dataset.absolute_to_relative_idx = None
    cfg = SimpleNamespace(
        trainable_config=SimpleNamespace(),
        dataset=SimpleNamespace(streaming=False),
        resume=True,
        checkpoint_path=tmp_path,
        batch_size=2,
        seed=42,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
    )
    parallel_dims = SimpleNamespace(dp_world_size=3, dp_rank=2, device_type="cpu")
    monkeypatch.setattr(training, "load_training_metadata", lambda _: {"dp_world_size": 3, "batch_size": 2})
    raw, _ = training.make_dataloaders(cfg, dataset, None, step=3, parallel_dims=parallel_dims)
    sampler = raw.sampler
    prepared = prepare_data_loader(raw, num_processes=3, process_index=2, rng_types=[])
    _, _, reference = _prepare(10, 2, 3, 2)
    epochs = [[batch.tolist() for batch in reference] for _ in range(3)]
    expected = epochs[1][1:] + epochs[2]
    actual = [
        batch.tolist()
        for batch in islice(training._cycle_training_dataloader(prepared, sampler), len(expected))
    ]
    assert actual == expected


def test_first_epoch_wrapper_is_released_before_regular_epochs(monkeypatch):
    skip = training.skip_first_batches
    wrappers = []

    def track_wrapper(*args):
        wrapper = skip(*args)
        wrappers.append(weakref.ref(wrapper))
        return wrapper

    monkeypatch.setattr(training, "skip_first_batches", track_wrapper)
    _, sampler, loader = _prepare(10, 2, 1, 0, {"epoch": 1, "start_index": 4})
    iterator = training._cycle_training_dataloader(loader, sampler)
    # Three remaining batches finish the resumed epoch, then the regular loader starts.
    list(islice(iterator, 4))
    assert len(wrappers) == 1
    assert wrappers[0]() is None
    iterator.close()


def test_streaming_loader_cycles_without_sampler_resume_handling():
    class Stream(IterableDataset):
        def __iter__(self):
            yield from range(3)

    loader = DataLoader(Stream(), batch_size=2)
    actual = list(islice(training._cycle_training_dataloader(loader, loader.sampler), 3))
    assert [batch.tolist() for batch in actual] == [[0, 1], [2], [0, 1]]
