import asyncio
import functools
import http.server
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.download_datasets import (
    EMPTY,
    RemoteDataset,
    RowRange,
    collect_remote_datasets,
    mirror_store,
    needed_rows,
    stage_windows,
)
from weathergen.utils.cli import get_download_parser

HOUR = np.timedelta64(1, "h")
EPOCH = np.datetime64("1970-01-01T00:00:00")
OBS_STREAM = {"name": "OBS", "type": "obs", "filenames": ["obs.zarr"], "geoinfo_channels": []}
ANEMOI_STREAM = {
    "name": "ERA5",
    "type": "anemoi",
    "filenames": ["era5.zarr"],
    "geoinfo_channels": [],
}


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def http_server(tmp_path):
    """Serve tmp_path over HTTP, without directory listings like a data catalog."""
    handler = functools.partial(_QuietHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def make_obs_store(path: Path, zarr_format: int = 2) -> None:
    """Observations of 6 hours, 2 rows per hour, in chunks of 4 rows."""
    group = zarr.create_group(store=path, zarr_format=zarr_format)
    data = group.create_array(name="data", shape=(12, 3), chunks=(4, 3), dtype="float32")
    data[:] = np.arange(36, dtype="float32").reshape(12, 3)
    data.attrs.update({"colnames": ["lat", "lon", "obsvalue_1"], "means": [0] * 3, "vars": [1] * 3})
    dates = group.create_array(name="dates", shape=(12, 1), chunks=(4, 1), dtype="datetime64[ns]")
    dates[:] = (EPOCH + np.repeat(np.arange(6), 2) * HOUR).reshape(12, 1)
    index = group.create_array(name="idx_197001010000_1", shape=(7,), chunks=(7,), dtype="int64")
    index[:] = np.arange(0, 14, 2)


def make_anemoi_store(path: Path) -> None:
    """Gridded anemoi dataset of 8 time steps of 6h from 2020-01-01, one step per chunk."""
    group = zarr.create_group(store=path, zarr_format=2)
    shape = (8, 2, 1, 3)
    data = group.create_array(name="data", shape=shape, chunks=(1, *shape[1:]), dtype="float32")
    data[:] = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    dates = group.create_array(name="dates", shape=(8,), chunks=(2,), dtype="datetime64[s]")
    dates[:] = np.datetime64("2020-01-01") + np.arange(8) * 6 * HOUR
    for name, size in [("latitudes", 3), ("longitudes", 3)] + [
        (s, 2) for s in ("mean", "stdev", "minimum", "maximum")
    ]:
        array = group.create_array(name=name, shape=(size,), chunks=(size,), dtype="float64")
        array[:] = np.arange(size) + 1
    group.attrs.update(
        {
            "frequency": "6h",
            "variables": ["t", "z"],
            "resolution": "test",
            "field_shape": [3],
            "flatten_grid": True,
            "ensemble_dimension": 1,
            "start_date": "2020-01-01T00:00:00",
            "end_date": "2020-01-02T18:00:00",
            "missing_dates": [],
        }
    )


def _files(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}


def _window(start, end, length=HOUR, step=HOUR) -> TimeWindowHandler:
    return TimeWindowHandler(np.datetime64(start), np.datetime64(end), length, step)


def test_row_range_union():
    assert RowRange(2, 6).union(RowRange(8, 12)) == RowRange(2, 12)
    assert RowRange(2, 6).union(EMPTY) == RowRange(2, 6)
    assert EMPTY.union(RowRange(2, 6)) == RowRange(2, 6)
    assert len(EMPTY) == 0


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_mirror_obs_rows(tmp_path, http_server, zarr_format):
    make_obs_store(tmp_path / "obs.zarr", zarr_format)
    target = tmp_path / "local" / "obs.zarr"
    arrays = ["data", "dates", "idx_197001010000_1", "not_there"]
    stats = asyncio.run(
        mirror_store(f"{http_server}/obs.zarr", target, arrays, RowRange(2, 6), transfers=2)
    )
    assert (stats.copied, stats.skipped, stats.missing) == (5, 0, 0)

    files = _files(target)
    if zarr_format == 2:
        chunks = {"data/0.0", "data/1.0", "dates/0.0", "dates/1.0", "idx_197001010000_1/0"}
        assert files == chunks | {".zgroup", ".zattrs"} | {
            f"{a}/{m}" for a in arrays[:3] for m in (".zarray", ".zattrs")
        }
    else:
        chunks = {"data/c/0/0", "data/c/1/0", "dates/c/0/0", "dates/c/1/0"}
        assert files == chunks | {"zarr.json", "idx_197001010000_1/c/0"} | {
            f"{a}/zarr.json" for a in arrays[:3]
        }

    local = zarr.open_group(target, mode="r")
    source = zarr.open_group(tmp_path / "obs.zarr", mode="r")
    np.testing.assert_array_equal(local["data"][2:6], source["data"][2:6])
    np.testing.assert_array_equal(local["idx_197001010000_1"][:], source["idx_197001010000_1"][:])
    assert dict(local["data"].attrs) == dict(source["data"].attrs)

    # a second run copies nothing
    stats = asyncio.run(mirror_store(f"{http_server}/obs.zarr", target, arrays, RowRange(2, 6)))
    assert (stats.copied, stats.skipped) == (0, 5)


def test_mirror_anemoi_rows(tmp_path, http_server):
    make_anemoi_store(tmp_path / "era5.zarr")
    target = tmp_path / "local" / "era5.zarr"
    arrays = ["data", "dates", "latitudes", "mean", "count"]
    asyncio.run(mirror_store(f"{http_server}/era5.zarr", target, arrays, RowRange(1, 4)))

    chunks = {f for f in _files(target) if "." not in f.rsplit("/", 1)[-1] or f.endswith(".0.0.0")}
    assert chunks == {
        "data/1.0.0.0",
        "data/2.0.0.0",
        "data/3.0.0.0",
        "dates/0",
        "dates/1",
        "latitudes/0",
        "mean/0",
    }
    local = zarr.open_group(target, mode="r")
    source = zarr.open_group(tmp_path / "era5.zarr", mode="r")
    np.testing.assert_array_equal(local["data"][1:4], source["data"][1:4])
    np.testing.assert_array_equal(local["dates"][:4], source["dates"][:4])
    assert dict(local.attrs) == dict(source.attrs)


def test_mirror_all_rows_and_dry_run(tmp_path, http_server):
    make_anemoi_store(tmp_path / "era5.zarr")
    target = tmp_path / "local" / "era5.zarr"
    stats = asyncio.run(
        mirror_store(f"{http_server}/era5.zarr", target, ["data"], None, dry_run=True)
    )
    assert stats.copied == 0 and not target.exists()
    stats = asyncio.run(mirror_store(f"{http_server}/era5.zarr", target, ["data"], None))
    assert stats.copied == 8
    np.testing.assert_array_equal(
        zarr.open_group(target, mode="r")["data"][:],
        zarr.open_group(tmp_path / "era5.zarr", mode="r")["data"][:],
    )


def test_mirror_requires_data_array(tmp_path, http_server):
    zarr.create_group(store=tmp_path / "empty.zarr", zarr_format=2).create_array(
        name="other", shape=(2,), dtype="int64"
    )
    with pytest.raises(ValueError, match="no 'data' array"):
        asyncio.run(
            mirror_store(f"{http_server}/empty.zarr", tmp_path / "x", ["data", "other"], None)
        )


def test_needed_rows_obs(tmp_path, http_server):
    make_obs_store(tmp_path / "obs.zarr")
    url = f"{http_server}/obs.zarr"

    def dataset(*windows):
        info = OmegaConf.create(OBS_STREAM)
        return RemoteDataset("OBS", url, tmp_path / "x", info, [("test", w) for w in windows])

    assert dataset().arrays == ["data", "dates", "idx_197001010000_1"]
    # the reader's samples for hour 1 (rows 2-3) and for hours 1-2 (rows 2-5)
    assert needed_rows(dataset(_window(EPOCH + HOUR, EPOCH + 2 * HOUR))) == RowRange(2, 4)
    assert needed_rows(dataset(_window(EPOCH + HOUR, EPOCH + 3 * HOUR))) == RowRange(2, 6)
    # two stages are combined
    rows = needed_rows(
        dataset(_window(EPOCH, EPOCH + HOUR), _window(EPOCH + 4 * HOUR, EPOCH + 5 * HOUR))
    )
    assert rows == RowRange(0, 10)


def test_needed_rows_anemoi(tmp_path, http_server):
    make_anemoi_store(tmp_path / "era5.zarr")
    url = f"{http_server}/era5.zarr"

    def dataset(*windows):
        info = OmegaConf.create(ANEMOI_STREAM)
        return RemoteDataset("ERA5", url, tmp_path / "x", info, [("test", w) for w in windows])

    assert "stdev" in dataset().arrays
    window = _window("2020-01-01T06", "2020-01-01T12", 6 * HOUR, 6 * HOUR)
    assert needed_rows(dataset(window)) == RowRange(1, 3)
    window = _window("2020-01-01T06", "2020-01-02T12", 6 * HOUR, 6 * HOUR)
    assert needed_rows(dataset(window)) == RowRange(1, 7)
    # no overlap with the dataset
    window = _window("2021-01-01T00", "2021-01-02T00", 6 * HOUR, 6 * HOUR)
    assert needed_rows(dataset(window)) == EMPTY


def test_collect_remote_datasets(tmp_path, http_server):
    local_dir = tmp_path / "local"
    make_anemoi_store(local_dir / "local.zarr")
    cf = OmegaConf.create(
        {
            "data_paths": [str(local_dir), f"{http_server}"],
            "streams": {
                "OBS": OBS_STREAM,
                "ERA5": ANEMOI_STREAM,
                "LOCAL": {"name": "LOCAL", "type": "anemoi", "filenames": ["local.zarr"]},
                "FESOM": {"name": "FESOM", "type": "fesom", "filenames": ["fesom.zarr"]},
            },
        }
    )
    datasets = collect_remote_datasets(cf, [], tmp_path / "out")
    assert [(d.stream, d.type, d.url, d.target.name) for d in datasets] == [
        ("OBS", "obs", f"{http_server}/obs.zarr", "obs.zarr"),
        ("ERA5", "anemoi", f"{http_server}/era5.zarr", "era5.zarr"),
    ]


def test_stage_windows():
    cf = OmegaConf.create(
        {
            "training_config": {
                "start_date": "1979-01-01T00:00",
                "end_date": "2022-12-31T00:00",
                "time_window_len": "06:00:00",
                "time_window_step": "06:00:00",
                "losses": {"physical": {"type": "LossPhysical", "enabled": False}},
                "model_input": {"forecasting": {"masking_strategy": "forecast"}},
            },
            "validation_config": {"start_date": "2023-10-01T00:00", "end_date": "2023-12-31T00:00"},
            "test_config": {"start_date": "2023-03-01T00:00", "end_date": "2023-03-02T00:00"},
        }
    )
    windows = stage_windows(cf, ["training", "validation", "test"])
    assert [(stage, tw.t_start) for stage, tw in windows] == [
        ("train", np.datetime64("1979-01-01T00:00")),
        ("val", np.datetime64("2023-10-01T00:00")),
        ("test", np.datetime64("2023-03-01T00:00")),
    ]
    assert windows[2][1].t_end == np.datetime64("2023-03-02T00:00")
    assert all(tw.t_window_len == 6 * HOUR for _, tw in windows)


def test_download_parser():
    args = get_download_parser().parse_args(["--all", "--config", "a.yml"])
    assert args.from_run_id is None and args.all
    assert args.transfers == 16 and args.stages is None and args.data_dir is None
    assert args.config == [Path("a.yml")] and args.mini_epoch == -1
    assert not hasattr(args, "reuse_run_id")
