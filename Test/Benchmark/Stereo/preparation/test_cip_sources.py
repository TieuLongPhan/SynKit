"""Source-license fields and direct scientific-data comparison for CIP inputs."""

import json
from zipfile import BadZipFile

import pytest

from Experiment.Stereo.preparation import cip_sources


def test_prepared_inputs_match_the_original_archive_with_only_citation_repairs():
    result = cip_sources.verify_published_inputs()
    assert result["records"] == result["coordinate_records"] == 300
    assert result["scientific_fields_changed"] == 0
    assert result["bibliographic_fields_changed"] == 2


@pytest.mark.parametrize("component", ["archive", "metadata"])
def test_invalid_source_archive_or_license_is_rejected(component):
    archive = cip_sources.ARCHIVE_PATH.read_bytes()
    metadata = cip_sources.METADATA_PATH.read_bytes()
    if component == "archive":
        archive = b"not a zip archive"
    else:
        source = json.loads(metadata)
        source["license"]["name"] = "MIT"
        metadata = json.dumps(source).encode()
    with pytest.raises((ValueError, BadZipFile)):
        cip_sources.publisher_inputs(archive, metadata)
