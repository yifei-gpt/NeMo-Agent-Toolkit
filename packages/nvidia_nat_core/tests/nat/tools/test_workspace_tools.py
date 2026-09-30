# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The workspace tools, checked where they failed silently: the root they pick when nobody names
one, the census branch a large workspace always takes, and a link that leads out of the workspace."""
import pytest

from nat.tool import workspace_ops as ops
from nat.tool import workspace_tools as wt


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("NAT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.delenv("NAT_SANDBOX_URL", raising=False)
    return tmp_path


def test_root_is_never_the_process_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("NAT_WORKSPACE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert wt._root() != tmp_path.resolve()


def test_paths_fold_under_the_root(tmp_path):
    root = str(tmp_path)
    assert ops.resolve(root, "a.txt").parent == tmp_path.resolve()
    assert ops.resolve(root, f"{tmp_path.name}/a.txt").parent == tmp_path.resolve()
    assert ops.resolve(root, f"{ops.SANDBOX_ROOT}/a.txt").parent == tmp_path.resolve()
    with pytest.raises(ValueError):
        ops.resolve(root, "../../etc/passwd")


def test_census_still_names_the_root(tmp_path, monkeypatch):
    # Past the cap the listing becomes a folder census; the root went missing exactly there.
    monkeypatch.setattr(ops, "CENSUS_ROWS", 3)
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    listing = ops.listing(str(tmp_path), max_entries=2)
    assert ops.SANDBOX_ROOT in listing
    assert "too many to list" in listing


def test_listing_names_the_root_and_the_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    listing = ops.listing(str(tmp_path), max_entries=50)
    assert ops.SANDBOX_ROOT in listing and "a.txt" in listing


def test_a_link_out_of_the_workspace_is_refused(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "secret").write_text("kept")
    (tmp_path / "ws" / "leak").symlink_to(tmp_path / "secret")
    for call in (lambda: ops.read(str(tmp_path / "ws"), "leak"), lambda: ops.write(str(tmp_path / "ws"), "leak", "x")):
        with pytest.raises(ValueError):
            call()
    assert (tmp_path / "secret").read_text() == "kept"


def test_without_a_sandbox_the_tools_run_here(workspace):
    assert wt._op_here("write", path="d/a.txt", content="hi").startswith("wrote d/a.txt")
    assert wt._op_here("read", path="d/a.txt") == "hi"
    with pytest.raises(ValueError):
        wt._op_here("read", path="../../etc/passwd")
