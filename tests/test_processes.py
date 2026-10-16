import sys
import threading
import time

import pytest

from vlogflow.processes import Budget, Cancelled, ResourceLimit, Runner


def test_cancellation_terminates_only_owned_child(tmp_path):
    stop = threading.Event()
    runner = Runner(tmp_path, cancel=stop.is_set)
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    start = time.monotonic()
    with pytest.raises(Cancelled):
        runner.run([sys.executable, "-c", "import time; time.sleep(30)"])
    timer.join()
    assert time.monotonic() - start < 5


def test_preflight_does_not_launch_process_when_memory_insufficient(tmp_path):
    marker = tmp_path / "launched"
    with pytest.raises(ResourceLimit):
        Runner(tmp_path).run(
            [sys.executable, "-c", f"open({str(marker)!r},'w').write('bad')"],
            budget=Budget(min_start_gib=100000),
        )
    assert not marker.exists()
