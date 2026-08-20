# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The workspace tools, checked where they failed silently: the root they pick when nobody names
one, and the census branch a large workspace always takes."""
import pytest

from nat.tool import workspace_tools as wt


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("NAT_WORKSPACE_DIR", str(tmp_path))
    return tmp_path


def test_root_is_never_the_process_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("NAT_WORKSPACE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert wt._root() != tmp_path.resolve()


def test_paths_fold_under_the_root(workspace):
    assert wt._resolve("a.txt").parent == workspace
    assert wt._resolve(f"{workspace.name}/a.txt").parent == workspace
    with pytest.raises(ValueError):
        wt._resolve("../../etc/passwd")


def test_census_still_names_the_root(workspace, monkeypatch):
    # Past the cap the listing becomes a folder census; the absolute path went missing exactly there.
    monkeypatch.setattr(wt, "CENSUS_ROWS", 3)
    for i in range(5):
        (workspace / f"f{i}.txt").write_text("x")
    listing = wt._listing(max_entries=2)
    assert str(workspace) in listing
    assert "too many to list" in listing


def test_listing_names_the_root_and_the_files(workspace):
    (workspace / "a.txt").write_text("hello")
    listing = wt._listing(max_entries=50)
    assert str(workspace) in listing and "a.txt" in listing
