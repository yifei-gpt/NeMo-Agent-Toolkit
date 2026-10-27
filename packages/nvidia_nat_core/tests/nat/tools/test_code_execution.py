# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import multiprocessing
from urllib.parse import urljoin

import pytest
import requests
from pytest_httpserver import HTTPServer

from nat.tool.code_execution import code_sandbox
from nat.tool.code_execution.local_sandbox.local_sandbox_server import do_execute

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def set_mp_spawn():
    """
    By default this will attempt to use fork, however this can cause issues specifically since within the
    context of pytest there may be several file descriptors open that the child process inherits.
    """
    multiprocessing.set_start_method("spawn", force=True)


def test_client_init(uri: str = "http://localhost:6000"):
    sandbox = code_sandbox.get_sandbox("local", uri=uri)
    assert isinstance(sandbox, code_sandbox.LocalSandbox)
    assert sandbox.url == str(urljoin(uri, "execute"))

    uri = uri + "/"
    sandbox = code_sandbox.get_sandbox("local", uri=uri)
    assert isinstance(sandbox, code_sandbox.LocalSandbox)
    assert sandbox.url == str(urljoin(uri, "execute"))


async def test_handle_response(httpserver: HTTPServer):
    client = code_sandbox.get_sandbox("local", uri=httpserver.url_for("/execute"))
    httpserver.expect_request(
        "/execute",
        method="POST",
    ).respond_with_json({
        "process_status": "completed", "stdout": "Hello World", "stderr": ""
    })

    resp = await client.execute_code(generated_code='print("Hello World")')
    assert isinstance(resp, dict)
    assert resp == {"process_status": "completed", "stdout": "Hello World", "stderr": ""}


async def test_bad_response(httpserver: HTTPServer):
    client = code_sandbox.get_sandbox("local", uri="http://localhost:9999")

    # Test that connection error is raised when the service is unavailable
    with pytest.raises(requests.exceptions.ConnectionError):
        _ = await client.execute_code(generated_code='print("Hello World")')

    # Test for JSON parsing error
    client = code_sandbox.get_sandbox("local", uri=httpserver.url_for("/execute"))
    httpserver.expect_request(
        "/execute",
        method="POST",
    ).respond_with_data("""
                                      "process_status": "completed",
                                      "stdout": "Hello World",
                                      "stderr",  "",
                                   }""")

    resp = await client.execute_code(generated_code='print("Hello World")')
    assert resp.get("process_status") == "error"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").startswith("Unknown error")


async def test_code_gen(httpserver: HTTPServer):
    client = code_sandbox.get_sandbox("local", uri=httpserver.url_for("/execute"))
    httpserver.expect_request("/execute", method="POST").respond_with_handler(do_execute)

    # Execute simple code
    resp = await client.execute_code(generated_code='print("Hello World")')
    assert resp.get("process_status") == "completed"
    assert resp.get("stdout").rstrip() == "Hello World"
    assert resp.get("stderr") == ""

    # Check Timeout
    resp = await client.execute_code(generated_code="import time; time.sleep(0.2)", timeout_seconds=0.05)
    assert resp.get("process_status") == "timeout"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").rstrip() == "Timed out"

    # Check Exception
    resp = await client.execute_code(generated_code="print(1/0)")
    assert resp.get("process_status") == "error"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").startswith("Traceback")

    # Check invalid code
    resp = await client.execute_code(generated_code="124ijfmpoeqfmpew')")
    assert resp.get("process_status") == "error"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").startswith("Traceback")

    # Check handle code block
    resp = await client.execute_code(generated_code="""
```
import json


print(5+5)
```
""")
    assert resp.get("process_status") == "completed"
    assert resp.get("stdout").rstrip() == "10"
    assert resp.get("stderr") == ""


def test_the_sandbox_server_requires_its_token(monkeypatch):
    """/execute needs the token; status stays open."""
    import importlib

    from nat.tool.code_execution.local_sandbox import local_sandbox_server as server
    monkeypatch.setenv("NAT_SANDBOX_TOKEN", "s3cret")
    fresh = importlib.reload(server)
    try:
        client = fresh.app.test_client()
        body = {"generated_code": "print(1)", "timeout": 10, "language": "python"}
        assert client.post("/execute", json=body).status_code == 403
        assert client.post("/execute", json=body, headers={"X-Sandbox-Token": "wrong"}).status_code == 403
        assert client.get("/").status_code == 200
        assert "NAT_SANDBOX_TOKEN" not in __import__("os").environ
    finally:
        monkeypatch.delenv("NAT_SANDBOX_TOKEN", raising=False)
        importlib.reload(server)


async def test_the_client_sends_the_token(httpserver: HTTPServer, monkeypatch):
    monkeypatch.setenv("NAT_SANDBOX_TOKEN", "s3cret")
    httpserver.expect_request("/execute", headers={"X-Sandbox-Token": "s3cret"}).respond_with_json(
        {"process_status": "completed", "stdout": "1\n", "stderr": ""})
    client = code_sandbox.get_sandbox("local", uri=httpserver.url_for("/"))
    resp = await client.execute_code(generated_code="print(1)", timeout_seconds=5)
    assert resp["stdout"] == "1\n"


@pytest.mark.parametrize("code, status, shown", [
    ("print('hi')\nimport sys\nsys.exit(0)", "completed", "hi"),
    ("print('hi')\nimport sys\nsys.exit(3)", "error", "[exit 3]"),
    # As run_code sends it: run as __main__, where unittest.main finds its tests
    ("import runpy, tempfile\nf = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)\n"
     "f.write('import unittest\\nclass T(unittest.TestCase):\\n    def test_a(self):\\n"
     "        self.assertEqual(1, 2)\\nunittest.main(argv=[\"t\"])\\n')\nf.close()\n"
     "runpy.run_path(f.name, run_name='__main__')\n", "error", "FAIL: test_a"),
    ("import argparse\nargparse.ArgumentParser().parse_args(['--nope'])\n", "error", "unrecognized arguments"),
])
def test_an_exit_keeps_its_output(code, status, shown):
    """sys.exit, unittest.main, argparse end the script, not the sandbox."""
    from nat.tool.code_execution.local_sandbox.local_sandbox_server import execute_python
    result = execute_python(code, timeout=30)
    assert result.process_status == status
    assert shown in result.stdout + result.stderr
    assert "hi" in result.stdout or "hi" not in code


def test_a_timeout_ends_what_the_code_started(tmp_path):
    from nat.tool.code_execution.local_sandbox.local_sandbox_server import execute_python
    marker = tmp_path / "alive"
    code = (f"import subprocess\nsubprocess.Popen(['sh', '-c', 'sleep 3; touch {marker}'])\n"
            "import time\ntime.sleep(30)\n")
    assert execute_python(code, timeout=1).process_status == "timeout"
    import time
    time.sleep(4)
    assert not marker.exists()
