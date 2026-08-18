"""Format name -> reader instance, mirroring ``dump_dataset_schema.py``'s own
``FORMAT_INSPECTORS`` registry pattern so a new source format is always a
two-line addition here plus one new ``readers/<format>_reader.py``, never a
change to ``convert_dataset.py``'s dispatch logic.
"""
from __future__ import annotations

from readers.base import DatasetReader
from readers.hdf5_reader import Hdf5Reader
from readers.one_x_world_model_reader import OneXWorldModelReader
from readers.raw_image_json_reader import RawImageJsonReader
from readers.rlds_reader import RldsReader
from readers.robomimic_hdf5_reader import RobomimicHdf5Reader

READER_REGISTRY: dict[str, DatasetReader] = {
    "hdf5": Hdf5Reader(),
    "one_x_world_model": OneXWorldModelReader(),
    "robomimic_hdf5": RobomimicHdf5Reader(),
    "rlds": RldsReader(),
    "raw_image_json": RawImageJsonReader(),
}


def get_reader(format_name: str) -> DatasetReader:
    try:
        return READER_REGISTRY[format_name]
    except KeyError:
        raise ValueError(
            f"no reader registered for format {format_name!r}; available: {sorted(READER_REGISTRY)}"
        ) from None
