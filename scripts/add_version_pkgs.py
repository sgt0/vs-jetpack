import shutil
from pathlib import Path
from typing import Any

from hatchling.builders.config import BuilderConfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface[BuilderConfig]):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)

        version_file = root / "_version.py"

        self._version_files = list[Path]()

        for pkg in self.build_config.packages:
            dest = root / pkg / "_version.py"
            self._version_files.append(dest)

            shutil.copy2(version_file, dest)

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        for file in self._version_files:
            file.unlink()
