"""Durability tests for the spool.

These cover the invariants the README claims: producer data is copied, not
moved; the same artifact keeps one identity; quotas are enforced at admission
so the video lane cannot starve telemetry; and a restart cleans up whatever a
crash left behind.

Everything here is local. No network, no credentials, no transport.
"""

import hashlib
import os

import pytest

from vmodal_robot.contracts import ArtifactInput, RevisionInput, SpoolFull
from vmodal_robot.spool import Spool, SpoolConfig
from vmodal_robot.utils import str_sha256_file


def _artifact(directory, rel_path, payload, kind="telemetry"):
    """Write a real file and describe it the way a recorder would."""
    path = os.path.join(str(directory), rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return ArtifactInput(
        rel_path=rel_path,
        kind=kind,
        content_type="application/octet-stream",
        source_path=path,
        byte_length=len(payload),
        checksum=hashlib.sha256(payload).hexdigest(),
        source_clock="monotonic",
    )


def _revision(artifacts, source_revision="rev-1", manifest_path="/tmp/ready.json"):
    return RevisionInput(
        source_id="robot-1",
        dataset_key="dataset-a",
        source_format="lerobot",
        source_version="v3",
        destination="collection/stream",
        source_revision=source_revision,
        complete=True,
        artifacts=tuple(artifacts),
        manifest_path=manifest_path,
    )


def _spool(tmp_path, **overrides):
    config = SpoolConfig(root=os.path.join(str(tmp_path), "spool"), **overrides)
    return Spool(config)


def test_accept_copies_the_bytes_and_keeps_the_source(tmp_path):
    """Admission copies into the spool and leaves the producer's file alone."""
    spool = _spool(tmp_path)
    try:
        source = _artifact(tmp_path / "recorder", "data/telemetry.bin", b"hello world")
        revision = spool.accept(_revision([source]))

        assert len(revision.artifact_ids) == 1
        artifact_id = revision.artifact_ids[0]
        artifact = spool.get_artifact(artifact_id)

        assert spool.artifact_state(artifact_id) == "READY"
        assert os.path.isfile(artifact.local_ref)
        assert str_sha256_file(artifact.local_ref) == source.checksum
        assert os.path.isfile(source.source_path), "the source must never be moved"
        assert spool.pending_count() == 1
    finally:
        spool.close()


def test_the_same_revision_twice_does_not_duplicate(tmp_path):
    """Identity is dataset + path + checksum, so a replay is a no-op."""
    spool = _spool(tmp_path)
    try:
        source = _artifact(tmp_path / "recorder", "data/telemetry.bin", b"same bytes")
        revision = _revision([source])

        first = spool.accept(revision)
        second = spool.accept(revision)

        assert first.revision_id == second.revision_id
        assert first.artifact_ids == second.artifact_ids
        assert spool.pending_count() == 1
        assert spool.status()["record_count"] == 1
    finally:
        spool.close()


def test_record_quota_is_enforced_at_admission(tmp_path):
    """Going over max_records is refused, and the refusal is recorded."""
    spool = _spool(tmp_path, max_records=1)
    try:
        first = _artifact(tmp_path / "recorder", "data/one.bin", b"first")
        spool.accept(_revision([first], source_revision="rev-1"))

        second = _artifact(tmp_path / "recorder", "data/two.bin", b"second")
        with pytest.raises(SpoolFull):
            spool.accept(_revision([second], source_revision="rev-2"))

        assert spool.pending_count() == 1
        assert spool.status()["rejected_handoffs"] == 1, "rejections must be auditable"
    finally:
        spool.close()


def test_video_cannot_eat_the_aux_reserve(tmp_path):
    """video_limit = max_bytes - aux_reserve_bytes, checked before any write."""
    spool = _spool(tmp_path, max_bytes=10_000, aux_reserve_bytes=9_000)
    try:
        # The video lane may only use 1_000 bytes, so 2_000 must be refused.
        video = _artifact(tmp_path / "recorder", "video/cam0.mp4", b"x" * 2_000, kind="video")
        with pytest.raises(SpoolFull):
            spool.accept(_revision([video]))

        # Telemetry of the same size still fits, because it uses the reserve.
        telemetry = _artifact(tmp_path / "recorder", "data/telemetry.bin", b"y" * 2_000)
        spool.accept(_revision([telemetry], source_revision="rev-2"))

        assert spool.status()["video_bytes"] == 0
        assert spool.pending_count() == 1
    finally:
        spool.close()


def test_restart_clears_what_a_crash_left_behind(tmp_path):
    """os_recover() drops orphan payloads and half-written temp files."""
    root = os.path.join(str(tmp_path), "spool")
    spool = Spool(SpoolConfig(root=root))
    source = _artifact(tmp_path / "recorder", "data/telemetry.bin", b"keep me")
    revision = spool.accept(_revision([source]))
    real_payload = spool.get_artifact(revision.artifact_ids[0]).local_ref
    spool.close()

    objects = os.path.join(root, "objects")
    orphan_dir = os.path.join(objects, "zz", "z" * 64)
    os.makedirs(orphan_dir, exist_ok=True)
    orphan = os.path.join(orphan_dir, "payload")
    with open(orphan, "wb") as handle:
        handle.write(b"orphan from a crashed run")
    stray = os.path.join(objects, ".accept-halfwritten")
    with open(stray, "wb") as handle:
        handle.write(b"interrupted copy")

    reopened = Spool(SpoolConfig(root=root))  # __init__ runs os_recover()
    try:
        assert not os.path.exists(orphan), "an unknown payload must be removed"
        assert not os.path.exists(stray), "a .accept- temp file must be removed"
        assert os.path.isfile(real_payload), "a known payload must survive"
        assert reopened.pending_count() == 1
    finally:
        reopened.close()


def test_status_reports_the_fields_operators_rely_on(tmp_path):
    """status() is the operator surface, so its shape is part of the contract."""
    spool = _spool(tmp_path)
    try:
        status = spool.status()
        for key in (
            "root",
            "spool_bytes",
            "record_count",
            "disk_free_bytes",
            "disk_headroom_bytes",
            "video_bytes",
            "lanes",
            "blocked_count",
            "retry_count",
            "rejected_handoffs",
            "last_receipt",
        ):
            assert key in status, f"status() lost the {key} field"

        assert set(status["lanes"]) == {"video", "aux"}
        assert status["record_count"] == 0
        assert status["last_receipt"] is None
        assert status["disk_free_bytes"] > 0
    finally:
        spool.close()
