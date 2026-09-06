"""Public search API for the AfyaPlus image retrieval index."""

import importlib.util
from pathlib import Path

INDEX_MODULE_PATH = Path(__file__).resolve().parent / 'week_5 ' / 'build_index.py'
_spec = importlib.util.spec_from_file_location('afya_image_index', INDEX_MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f'Unable to load image index module: {INDEX_MODULE_PATH}')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

build_index = _module.build_index
search = _module.search

__all__ = ['build_index', 'search']