"""Runtime monitoring helpers."""

import os
import subprocess
import threading


def start_gpu_monitor(self):
    """Start a background GPU utilization monitor."""
    if not self.gpu_monitor:
        return None, None
    stop_event = threading.Event()

    def _worker():
        while not stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if out:
                    parts = [p.strip() for p in out.split(",")]
                    if len(parts) >= 2:
                        self._gpu_status = f"{parts[0]}%/{parts[1]}%"
                    else:
                        self._gpu_status = out
            except Exception:
                pass
            stop_event.wait(self.gpu_monitor_interval)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, stop_event


def start_ram_monitor(self):
    """Start a background RAM usage monitor."""
    if not self.ram_monitor:
        return None, None
    stop_event = threading.Event()

    def _read_rss_mb():
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            return proc.memory_info().rss / (1024 ** 2)
        except Exception:
            try:
                with open(f"/proc/{os.getpid()}/statm", "r") as f:
                    parts = f.read().strip().split()
                if len(parts) >= 2:
                    rss_pages = int(parts[1])
                    return rss_pages * (os.sysconf("SC_PAGE_SIZE") / (1024 ** 2))
            except Exception:
                return None
        return None

    def _worker():
        while not stop_event.is_set():
            rss_mb = _read_rss_mb()
            if rss_mb is not None:
                self._ram_status = f"{rss_mb/1024:.2f}GB"
            stop_event.wait(self.ram_monitor_interval)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, stop_event
