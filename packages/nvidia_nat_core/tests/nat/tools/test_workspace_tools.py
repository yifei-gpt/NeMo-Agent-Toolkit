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
    scratch = wt._root()
    scratch.rmdir()
    assert scratch != tmp_path.resolve()


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
    ws = str(tmp_path / "ws")
    for call in (lambda: ops.read(ws, "leak", max_chars=100), lambda: ops.write(ws, "leak", "x")):
        with pytest.raises(ValueError):
            call()
    assert (tmp_path / "secret").read_text() == "kept"


@pytest.mark.parametrize("text, encoding", [
    ("name,city\nRésumé,Zürich\n", "latin-1"), ("“quoted” – €100\n", "cp1252"), ("名前,42\nCafé,7\n", "utf-16"),
    ("\ufeffid,Zoë\n", "utf-8"), ("中文测试：季度报告\n", "gb18030"), ("你好", "gb18030"),
    ("日本語のテキストです。\n", "shift_jis")])
def test_a_file_reads_back_in_its_own_encoding(tmp_path, monkeypatch, text, encoding):
    monkeypatch.setattr(ops, "_CACHE", tmp_path / "cache")
    (tmp_path / "f.txt").write_bytes(text.encode(encoding))
    assert ops.read(str(tmp_path), "f.txt", max_chars=100).split("\n\n[not UTF-8")[0] == text.lstrip("\ufeff")


def test_without_a_sandbox_the_tools_run_here(workspace):
    assert wt._op_here("write", path="d/a.txt", content="hi").startswith("wrote d/a.txt")
    assert wt._op_here("read", path="d/a.txt", max_chars=100) == "hi"
    with pytest.raises(ValueError):
        wt._op_here("read", path="../../etc/passwd", max_chars=100)


def test_a_real_folder_named_like_the_root_is_not_folded(tmp_path):
    # In the sandbox the root is /workspace, so `workspace/x` is folded to x
    root = tmp_path / "workspace"
    (root / "workspace").mkdir(parents=True)
    (root / "notes.txt").write_text("outer")
    (root / "workspace" / "notes.txt").write_text("inner")
    assert ops.read(str(root), "workspace/notes.txt", max_chars=100) == "inner"
    ops.write(str(root), "workspace/notes.txt", "changed")
    assert (root / "notes.txt").read_text() == "outer"
    assert ops.read(str(root), "workspace/missing/../notes.txt", max_chars=100) == "changed"


def test_an_edit_keeps_crlf(tmp_path):
    (tmp_path / "w.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    assert ops.edit(str(tmp_path), "w.txt", "b\nc", "B\nC").startswith("edited")
    assert (tmp_path / "w.txt").read_bytes() == b"a\r\nB\r\nC\r\n"


@pytest.fixture
def bridged(tmp_path, monkeypatch):
    """Local bash as the task container, read through a text pipe like the bridge."""
    import subprocess
    app = tmp_path / "app"
    app.mkdir()

    def sh(command, timeout=60.0):
        done = subprocess.run(["bash", "-c", command], cwd=app, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace",
                              timeout=timeout, check=False)
        return True, done.stdout

    monkeypatch.setattr(wt, "_bridge", lambda: "http://bridge")
    monkeypatch.setattr(wt, "_sh", sh)
    return app


async def _call(register, config, **kwargs):
    async with register(config, None) as info:
        return await info.single_fn(info.input_schema(**kwargs))


async def test_a_bridged_edit_keeps_bytes_it_does_not_touch(bridged):
    (bridged / "w.txt").write_bytes(b"a\r\nb\r\n")
    (bridged / "latin.txt").write_bytes("caf\xe9 b\n".encode("latin-1"))
    said = await _call(wt.edit_file, wt.WorkspaceEditConfig(), path="w.txt", old="b", new="B")
    assert said.startswith("edited") and (bridged / "w.txt").read_bytes() == b"a\r\nB\r\n"
    said = await _call(wt.edit_file, wt.WorkspaceEditConfig(), path="latin.txt", old="b", new="B")
    assert "not UTF-8" in said and (bridged / "latin.txt").read_bytes() == "caf\xe9 b\n".encode("latin-1")


async def test_a_bridged_read_pages_by_bytes(bridged):
    text = "é" * 30 + "end"
    (bridged / "u.txt").write_text(text, encoding="utf-8")
    config, got, offset = wt.WorkspaceReadConfig(max_chars=7), "", 0
    for _ in range(20):
        part = await _call(wt.read_file, config, path="u.txt", offset=offset)
        chunk, _, rest = part.partition("\n... ")
        got += chunk
        if not rest:
            break
        offset = int(rest.rsplit("offset=", 1)[1])
    assert got == text


async def test_a_bridged_listing_keeps_a_hidden_folder(bridged):
    (bridged / ".github").mkdir()
    (bridged / ".github" / "ci.yml").write_text("x")
    listing = await _call(wt.list_directory, wt.WorkspaceListConfig(), subdir=".github")
    assert ".github/ci.yml" in listing
