"""Per-process event log: one dated file per stem under aiva/logs/.

The console has no scrollback once its window closes — these files are what 'check the logs'
reads when something odd happened hours ago. Each process writes its own stem so the house and
each sister can be diagnosed separately even though they run at the same time:
  aiva/logs/aiva-YYYYMMDD.log   (Aiva's process — unchanged name from before the house)
  aiva/logs/saga-YYYYMMDD.log
  aiva/logs/house-YYYYMMDD.log
"""

import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


class EventLog:
    """Callable: `log = EventLog("saga"); log("AWAKE — ...")`. Never raises (logging must never
    take her down)."""

    def __init__(self, stem: str, log_dir: str = LOG_DIR):
        self.stem = stem
        self.log_dir = log_dir

    def path(self) -> str:
        return os.path.join(self.log_dir, f"{self.stem}-{datetime.now():%Y%m%d}.log")

    def __call__(self, msg: str):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self.path(), "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%H:%M:%S} {msg}\n")
        except OSError:
            pass
