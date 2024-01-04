import logging
import os
import shutil
import tempfile
from pathlib import Path

import pydantic


class TempDir(pydantic.BaseModel):
    """A helper to centralize handling and cleanup of dirs"""

    path: Path = pydantic.Field(
        default_factory=lambda: Path(tempfile.mkdtemp())
    )

    def __del__(self) -> None:
        try:
            self._cleanup()
        except AttributeError:
            pass

    def _cleanup(self) -> None:
        # path may be deleted if files were moved elsewhere
        if not self.path.exists():
            return
        logging.warning(f"Removing {self.path}")
        for root, dirs, files in os.walk(self.path):
            for f in files + dirs:
                (Path(root) / f).chmod(0o700)
        shutil.rmtree(self.path)
