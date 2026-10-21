import os

import pytest

import weathergen.datasets.data_reader_obs as data_reader_obs
from weathergen.datasets.data_reader_obs import DataReaderObs


@pytest.fixture
def opened_stores(monkeypatch):
    """Record the stores opened by zarr.open instead of opening them."""
    opened = []

    def fake_open(filename, mode):
        opened.append(filename)
        return {"data": f"data{len(opened)}", "dates": f"dates{len(opened)}"}

    monkeypatch.setattr(data_reader_obs.zarr, "open", fake_open)
    return opened


def _reader(filename) -> DataReaderObs:
    """Reader with an opened store, without the dataset its constructor needs."""
    reader = DataReaderObs.__new__(DataReaderObs)
    reader.filename = filename
    reader._open_store()
    return reader


def test_open_store(opened_stores):
    reader = _reader("https://example.com/data.zarr")

    assert opened_stores == ["https://example.com/data.zarr"]
    assert reader.data == "data1"
    assert reader.dt == "dates1"
    assert reader._store_pid == os.getpid()


def test_reopen_remote_store_after_fork(opened_stores):
    reader = _reader("https://example.com/data.zarr")
    # as in a forked data loader worker
    reader._store_pid = os.getpid() + 1

    reader._reopen_store_after_fork()

    assert opened_stores == ["https://example.com/data.zarr"] * 2
    assert reader.data == "data2"
    assert reader.dt == "dates2"
    assert reader._store_pid == os.getpid()

    # only once per process
    reader._reopen_store_after_fork()
    assert len(opened_stores) == 2


def test_no_reopen_in_same_process(opened_stores):
    reader = _reader("https://example.com/data.zarr")

    reader._reopen_store_after_fork()

    assert len(opened_stores) == 1
    assert reader.data == "data1"


def test_no_reopen_local_store_after_fork(opened_stores, tmp_path):
    reader = _reader(tmp_path / "data.zarr")
    reader._store_pid = os.getpid() + 1

    reader._reopen_store_after_fork()

    assert len(opened_stores) == 1
    assert reader.data == "data1"
