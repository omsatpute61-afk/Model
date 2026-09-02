"""Dataset construction: manifests, field augmentation, synthetic images.

Requires torch. The field device never imports this package - see
``cropguard.edge`` for the inference-side preprocessing.
"""

from .manifest import (
    Record,
    manifest_summary,
    read_manifest,
    scan_image_folder,
    stratified_group_split,
    write_manifest,
)

__all__ = [
    "Record",
    "scan_image_folder",
    "stratified_group_split",
    "write_manifest",
    "read_manifest",
    "manifest_summary",
]
