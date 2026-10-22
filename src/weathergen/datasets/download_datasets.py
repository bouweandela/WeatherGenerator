# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Download the remote datasets of a run, so that it can run on nodes without internet access.

Datasets that resolve to a URL (a URL in the stream's `filenames` or in `data_paths`) are
mirrored to a local directory, chunk by chunk and byte for byte, restricted to the rows that
the readers access for the dates of the run's stages: `training_config`/`validation_config`
for training, `test_config` for inference. The rows come from the readers themselves, opened
on the remote dataset. Metadata, statistics and index arrays are the ones of the full remote
dataset, so the local copy behaves exactly like the remote one for those dates; chunks outside
the needed rows are simply absent (zarr returns the fill value for them).

Usage, with the same arguments as `train`/`inference` (`uv run download_datasets --help`):

    WEATHERGEN_PRIVATE_CONF=... uv run download_datasets --from-run-id mz7oyrr5 \
        --config config/jsc_catalog_inference.yml --data-dir /scratch/datasets

Then put the local directory *before* the URL in `data_paths` (private config or
`--options`), since the first `data_paths` entry that has the dataset wins.
"""

import argparse
import asyncio
import itertools
import logging
import math
import ssl
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import certifi
import fsspec.config
import numpy as np
import pandas as pd
import zarr
import zarr.api.asynchronous as zarr_async
from omegaconf import OmegaConf
from zarr.abc.store import Store
from zarr.core.array import AsyncArray
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.group import AsyncGroup
from zarr.storage import FsspecStore, LocalStore

from weathergen.common import config
from weathergen.common.config import Config, str_to_datetime64
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import TimeWindowHandler, is_remote_path
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.datasets.multi_stream_data_sampler import _resolve_data_path
from weathergen.utils import cli

_logger = logging.getLogger(__name__)

# Arrays the readers access when a store has no consolidated metadata (an HTTP server does
# not list directories, so the array names cannot be discovered).
ANEMOI_ARRAYS = [
    "data",
    "dates",
    "latitudes",
    "longitudes",
    "mean",
    "stdev",
    "minimum",
    "maximum",
    "count",
    "sums",
    "squares",
    "has_nans",
]
OBS_BASE_DATETIME = "1970-01-01T00:00:00"
# the readers' stage names for the config stages
READER_STAGES = {"training": "train", "validation": "val", "test": "test"}
_RETRIES = 3


@dataclass(frozen=True)
class RowRange:
    """Rows [start, end) along the first axis of the time-indexed arrays."""

    start: int
    end: int

    def __post_init__(self):
        assert 0 <= self.start <= self.end, (self.start, self.end)

    def __len__(self) -> int:
        return self.end - self.start

    def union(self, other: "RowRange") -> "RowRange":
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        return RowRange(min(self.start, other.start), max(self.end, other.end))


EMPTY = RowRange(0, 0)


@dataclass
class RemoteDataset:
    stream: str
    url: str
    target: Path
    stream_info: Config
    windows: list[tuple[str, TimeWindowHandler]]  # (reader stage, dates) per config stage

    @property
    def type(self) -> str:
        return self.stream_info["type"]

    @property
    def arrays(self) -> list[str]:
        if self.type == "obs":
            base = pd.Timestamp(self.stream_info.get("base_datetime", OBS_BASE_DATETIME))
            return ["data", "dates", f"idx_{base.strftime('%Y%m%d%H%M')}_1"]
        return ANEMOI_ARRAYS


@dataclass
class MirrorStats:
    copied: int = 0
    skipped: int = 0
    missing: int = 0
    nbytes: int = 0

    def __str__(self) -> str:
        return (
            f"{self.copied} chunks ({self.nbytes / 2**30:.2f} GB) copied, "
            f"{self.skipped} already present, {self.missing} not in source"
        )


def needed_rows(dataset: RemoteDataset) -> RowRange:
    """
    Rows the readers access for the dataset's windows, from the readers opened on the remote
    dataset: the sample index of the obs reader, the subset dates of the anemoi reader.
    """
    rows = EMPTY
    for stage, tw_handler in dataset.windows:
        if dataset.type == "obs":
            reader = DataReaderObs(tw_handler, dataset.url, dataset.stream_info, stage)
            if reader.length() > 0:
                start, end = reader.indices_start.min(), reader.indices_end.max() + 1
                rows = rows.union(RowRange(int(start), max(int(end), int(start))))
        else:
            reader = DataReaderAnemoi(tw_handler, dataset.url, dataset.stream_info, stage)
            if reader.ds is not None and reader.length() > 0:
                dates = zarr.open_group(dataset.url, mode="r")["dates"][:]
                start = np.searchsorted(dates, reader.ds.dates[0])
                end = np.searchsorted(dates, reader.ds.dates[-1], side="right")
                rows = rows.union(RowRange(int(start), int(end)))
    return rows


def _chunk_keys(array: AsyncArray, rows: RowRange | None) -> Iterator[str]:
    """Store keys of the chunks of `array` covering `rows` (all chunks if rows is None)."""
    chunk_shape = array.metadata.chunk_grid.chunk_shape
    nchunks = [math.ceil(s / c) for s, c in zip(array.shape, chunk_shape, strict=True)]
    if rows is None:
        first = range(nchunks[0])
    else:
        first = range(rows.start // chunk_shape[0], math.ceil(rows.end / chunk_shape[0]))
    prefix = f"{array.path}/" if array.path else ""
    for coords in itertools.product(first, *[range(n) for n in nchunks[1:]]):
        yield prefix + array.metadata.encode_chunk_key(coords)


def _chunk_nbytes(array: AsyncArray) -> int:
    return math.prod(array.metadata.chunk_grid.chunk_shape) * array.dtype.itemsize


def open_store(location: str | Path) -> Store:
    if is_remote_path(location):
        # a new filesystem per store: fsspec's cached instance keeps an HTTP session bound to
        # the event loop that first used it, which is closed after `asyncio.run`.
        options = {"skip_instance_cache": True}
        return FsspecStore.from_url(str(location), storage_options=options)
    return LocalStore(Path(location))


async def close_store(store: Store) -> None:
    """Close the HTTP session of a remote store (fsspec keeps it open otherwise)."""
    session = getattr(getattr(store, "fs", None), "_session", None)
    if session is not None:
        await session.close()


def use_certifi_ca_bundle() -> None:
    """
    Verify HTTPS servers against certifi's CA bundle, for this tool and the readers alike: the
    OpenSSL default bundle that aiohttp uses can be outdated (macOS lacks newer roots).
    """
    context = ssl.create_default_context(cafile=certifi.where())
    fsspec.config.conf.setdefault("https", {})["ssl"] = context


async def _array_names(group: AsyncGroup, candidates: list[str]) -> list[str]:
    """Arrays of the group: from the consolidated metadata if there is one, else those of
    `candidates` that exist."""
    consolidated = group.metadata.consolidated_metadata
    if consolidated is not None:
        return [
            name
            for name, meta in consolidated.flattened_metadata.items()
            if not isinstance(meta, type(group.metadata))
        ]
    names = []
    for name in candidates:
        try:
            await group.getitem(name)
        except KeyError:
            _logger.info(f"{group.store}: no array '{name}', skipped.")
            continue
        names.append(name)
    return names


async def _get_with_retries(store: Store, key: str) -> Buffer | None:
    for attempt in range(_RETRIES):
        try:
            return await store.get(key, prototype=default_buffer_prototype())
        except Exception as e:  # noqa: BLE001 network errors come in many types
            if attempt == _RETRIES - 1:
                raise
            _logger.warning(f"Retrying {key} after error: {e}")
            await asyncio.sleep(2**attempt)
    return None


async def _copy_keys(
    source: Store,
    target: Store,
    keys: Iterator[str],
    transfers: int,
    overwrite: bool,
    stats: MirrorStats,
    batch_size: int = 1024,
) -> None:
    semaphore = asyncio.Semaphore(transfers)
    last_log = time.monotonic()

    async def copy_one(key: str) -> None:
        nonlocal last_log
        async with semaphore:
            if not overwrite and await target.exists(key):
                stats.skipped += 1
                return
            buffer = await _get_with_retries(source, key)
            if buffer is None:
                stats.missing += 1
                return
            await target.set(key, buffer)
            stats.copied += 1
            stats.nbytes += len(buffer)
            if time.monotonic() - last_log > 10:
                last_log = time.monotonic()
                _logger.info(f"... {stats}")

    for batch in itertools.batched(keys, batch_size):
        await asyncio.gather(*(copy_one(key) for key in batch))


async def mirror_store(
    source: str | Path,
    target: str | Path,
    arrays: list[str],
    rows: RowRange | None,
    transfers: int = 16,
    overwrite: bool = False,
    dry_run: bool = False,
) -> MirrorStats:
    """
    Copy the zarr store at `source` to `target`: metadata and arrays byte for byte, restricted
    along the first axis to `rows` for the arrays with as many rows as `data` (all rows when
    `rows` is None). `arrays` are the arrays to look for when the store has no consolidated
    metadata. Existing chunks are kept unless `overwrite` is set, so a copy can be resumed.
    """
    src = open_store(source)
    dst = open_store(target)
    try:
        return await _mirror_store(
            src, dst, arrays, rows, transfers, overwrite, dry_run, label=f"{source} -> {target}"
        )
    finally:
        await close_store(src)


async def _mirror_store(
    src, dst, arrays, rows, transfers, overwrite, dry_run, label: str
) -> MirrorStats:
    prototype = default_buffer_prototype()
    group = await zarr_async.open_group(store=src, mode="r")
    stats = MirrorStats()

    names = await _array_names(group, arrays)
    if "data" not in names:
        raise ValueError(f"{label}: no 'data' array, not a WeatherGenerator dataset.")
    data = await group.getitem("data")
    nrows = data.shape[0]

    def row_subset(array: AsyncArray) -> RowRange | None:
        return rows if (rows is not None and array.shape[0] == nrows) else None

    metadata = list(group.metadata.to_buffer_dict(prototype).items())
    chunk_keys = []
    estimate = 0
    for name in names:
        array = await group.getitem(name)
        if not isinstance(array, AsyncArray):
            continue
        prefix = f"{array.path}/" if array.path else ""
        for key, buffer in array.metadata.to_buffer_dict(prototype).items():
            metadata.append((prefix + key, buffer))
        array_keys = list(_chunk_keys(array, row_subset(array)))
        chunk_keys.append(iter(array_keys))
        estimate += len(array_keys) * _chunk_nbytes(array)
        _logger.info(
            f"{name}: shape {array.shape}, chunks {array.metadata.chunk_grid.chunk_shape}, "
            f"rows {row_subset(array) or 'all'} -> {len(array_keys)} chunks"
        )
    _logger.info(f"{label}: at most {estimate / 2**30:.2f} GB uncompressed to copy.")
    if dry_run:
        return stats

    for key, buffer in metadata:
        if buffer is not None:
            await dst.set(key, buffer)
    await _copy_keys(src, dst, itertools.chain(*chunk_keys), transfers, overwrite, stats)
    _logger.info(f"{label}: {stats}")
    return stats


def stage_windows(cf: Config, stages: list[str]) -> list[tuple[str, TimeWindowHandler]]:
    """Time windows of the requested stages, derived as in the Trainer: validation inherits
    from training and test from validation."""
    from weathergen.train.utils import (
        cfg_keys_to_filter,
        filter_config_by_enabled,
        get_active_stage_config,
    )

    cfgs = {}
    cfgs["training"] = filter_config_by_enabled(cf.training_config, cfg_keys_to_filter)
    cfgs["validation"] = get_active_stage_config(
        cfgs["training"], cf.get("validation_config", {}), cfg_keys_to_filter
    )
    cfgs["test"] = get_active_stage_config(
        cfgs["validation"], cf.get("test_config", {}), cfg_keys_to_filter
    )
    windows = []
    for stage in stages:
        mode_cfg = cfgs[stage]
        tw_handler = TimeWindowHandler(
            str_to_datetime64(mode_cfg.start_date),
            str_to_datetime64(mode_cfg.end_date),
            pd.Timedelta(mode_cfg.time_window_len).to_timedelta64(),
            pd.Timedelta(mode_cfg.time_window_step).to_timedelta64(),
        )
        _logger.info(f"{stage}: {tw_handler}")
        windows.append((READER_STAGES[stage], tw_handler))
    return windows


def collect_remote_datasets(
    cf: Config, windows: list[tuple[str, TimeWindowHandler]], data_dir: Path
) -> list[RemoteDataset]:
    """The datasets of the run's streams that resolve to a URL, with their local targets."""
    datasets = []
    data_paths = cf.get("data_paths", [])
    for stream_name, stream_info in cf.streams.items():
        for fname in stream_info.get("filenames", []):
            filename = _resolve_data_path(fname, data_paths)
            if filename is None:
                _logger.warning(f"{stream_name}: '{fname}' not found in {data_paths}, skipped.")
                continue
            if not is_remote_path(filename):
                _logger.info(f"{stream_name}: '{fname}' is local ({filename}), nothing to do.")
                continue
            if stream_info["type"] not in ("obs", "anemoi"):
                _logger.warning(
                    f"{stream_name}: downloading '{stream_info['type']}' datasets is not "
                    f"supported, copy {filename} by hand."
                )
                continue
            url = str(filename)
            datasets.append(
                RemoteDataset(
                    stream=stream_name,
                    url=url,
                    target=data_dir / url.rstrip("/").rsplit("/", 1)[-1],
                    stream_info=stream_info,
                    windows=windows,
                )
            )
    return datasets


async def download(datasets: list[RemoteDataset], args: argparse.Namespace) -> None:
    for dataset in datasets:
        _logger.info(f"{dataset.stream}: {dataset.url} -> {dataset.target}")
        rows = None if args.all else needed_rows(dataset)
        if rows is not None and len(rows) == 0:
            _logger.warning(f"{dataset.stream}: no data in the requested dates.")
        await mirror_store(
            dataset.url,
            dataset.target,
            dataset.arrays,
            rows,
            transfers=args.transfers,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )


def main(argl: list[str]) -> list[RemoteDataset]:
    args = cli.get_download_parser().parse_args(argl)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    use_certifi_ca_bundle()

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.from_run_id, args.mini_epoch, args.base_config, *args.config, cli_overwrite
    )
    if OmegaConf.is_missing(cf, "streams"):  # a new run: streams come from the directory
        cf.streams = config.load_streams(Path(cf.streams_directory))
    stages = args.stages or (["test"] if args.from_run_id else ["training", "validation"])
    windows = stage_windows(cf, stages)
    if args.start or args.end:
        windows = [
            (
                stage,
                TimeWindowHandler(
                    str_to_datetime64(args.start) if args.start else tw.t_start,
                    str_to_datetime64(args.end) if args.end else tw.t_end,
                    tw.t_window_len,
                    tw.t_window_step,
                ),
            )
            for stage, tw in windows
        ]
        for stage, tw in windows:
            _logger.info(f"{stage} with --start/--end: {tw}")

    data_paths = cf.get("data_paths", [])
    data_dir = args.data_dir
    if data_dir is None:
        local = [Path(p) for p in data_paths if not is_remote_path(p)]
        if not local:
            raise ValueError("No local directory in data_paths, pass --data-dir.")
        data_dir = local[0]
    data_dir = data_dir.resolve()

    datasets = collect_remote_datasets(cf, windows, data_dir)
    if not datasets:
        _logger.info("All datasets are local, nothing to download.")
        return datasets
    asyncio.run(download(datasets, args))

    suggested = [str(data_dir)] + [str(p) for p in data_paths if Path(str(p)) != data_dir]
    _logger.info(
        "Done. Put the local directory before the URL in data_paths so that the copies are "
        f"used, e.g. --options 'data_paths=[{', '.join(suggested)}]'"
    )
    return datasets


def download_datasets():
    """Entry point for `uv run download_datasets`."""
    main(sys.argv[1:])


if __name__ == "__main__":
    download_datasets()
