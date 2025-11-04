import os
import platform
import re
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from sysconfig import get_platform
from tempfile import gettempdir
from typing import Any
from urllib.parse import urljoin
from zipfile import ZipFile

import tomllib
from auditwheel.wheel_abi import analyze_wheel_abi
from hatchling.builders.config import BuilderConfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PARENT_DIR = Path(__file__).parent
REPO_ROOT = PARENT_DIR.parents[1]
LIBMVSR_SOURCE_DIR = REPO_ROOT / "mvsr"
SDIST_SOURCE_DIR = PARENT_DIR / "libmvsr"
TARGET_DIR = PARENT_DIR / "mvsr" / "lib"

PYPROJECT_TOML = PARENT_DIR / "pyproject.toml"

LIBRARY_EXTENSIONS = ["so", "dylib", "dll"]
BASE_URL = urljoin(
    tomllib.loads(PYPROJECT_TOML.read_text())["project"]["urls"]["Repository"] + "/", "tree/main/"
)


class CustomBuildHook(BuildHookInterface[BuilderConfig]):
    def initialize(self, version: str, build_data: dict[str, Any]):
        if not os.environ.get("CI"):
            print(f"building '{LIBMVSR_SOURCE_DIR.name}' library ...")
            self.build_library()

        build_data["tag"] = f"py3-none-{self.get_platform_tag()}"

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str):
        with ZipFile(artifact_path) as zip_file:
            zip_data = {info: zip_file.read(info) for info in zip_file.infolist()}

        with ZipFile(artifact_path, "w") as zip_file:
            for info, content in zip_data.items():
                if info.filename.endswith(".dist-info/METADATA"):
                    content = content.decode()
                    for match in reversed(list(FIND_CROSS_REFERENCES_REGEX.finditer(content))):
                        reference = (PARENT_DIR / match.group(1)).resolve()
                        fixed = urljoin(BASE_URL, str(reference.relative_to(REPO_ROOT)))
                        if fixed.rsplit(".", 1)[-1].lower() in {"jpg", "png", "svg", "webp"}:
                            fixed = fixed.replace("github.com/", "raw.githubusercontent.com/")
                            fixed = fixed.replace("/tree/", "/")
                        content = content[: match.start(1)] + fixed + content[match.end(1) :]

                zip_file.writestr(info, content)

    def build_library(self):
        is_sdist_build = SDIST_SOURCE_DIR.is_dir()
        source_dir = SDIST_SOURCE_DIR if is_sdist_build else LIBMVSR_SOURCE_DIR

        TARGET_DIR.mkdir(exist_ok=True)

        if shutil.which("nix") and not is_sdist_build:
            subprocess.check_call(["nix", "build"], cwd=source_dir)
            out_dir = source_dir / "result" / "lib"
        else:
            out_dir = source_dir / "build"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir()

            generator = ["-G", "Visual Studio 17 2022"] if platform.system() == "Windows" else []
            subprocess.check_call(
                ["cmake", "-B", out_dir.name, "-DTESTING=off", *generator],
                cwd=source_dir,
            )
            subprocess.check_call(
                ["cmake", "--build", out_dir.name, "--config", "Release"],
                cwd=source_dir,
            )

        copied_library = False
        for path in chain(*(out_dir.glob(f"**/*.{ext}") for ext in LIBRARY_EXTENSIONS)):
            if (target_path := TARGET_DIR / path.name).is_file():
                target_path.unlink()
            shutil.copy(path, target_path)
            copied_library = True

        if not copied_library:
            raise RuntimeError(
                f"failed to find library file ({', '.join(LIBRARY_EXTENSIONS)}) in '{out_dir}'"
            )

    def get_platform_tag(self):
        platform_tag = get_platform().replace("-", "_").replace(".", "_")

        if platform.system() == "Linux":
            platform_tag = platform_tag.replace("linux", "manylinux2014")
            dummy_wheel_path = Path(gettempdir()) / "dummy.whl"
            with ZipFile(dummy_wheel_path, "w", strict_timestamps=False) as zip_file:
                records_str = ""
                for lib_path in chain(*(TARGET_DIR.glob(f"*.{ext}") for ext in LIBRARY_EXTENSIONS)):
                    zip_file.write(lib_path, lib_path.name)
                    records_str += f"{lib_path.name},,\n"
                zip_file.writestr("dummy.dist-info/RECORD", records_str)
            wheel_info = analyze_wheel_abi(None, None, dummy_wheel_path, frozenset(), False, False)

            if wheel_info.external_refs[wheel_info.policies.lowest.name].libs:
                platform_tag = f"{wheel_info.overall_policy.name}_{platform_tag}"

        elif platform.system() == "Darwin":
            version, _, architecture = platform.mac_ver()
            platform_tag = f"macosx_{version.split('.')[0]}_0_{architecture}"

        return platform_tag


FIND_CROSS_REFERENCES_REGEX = re.compile(r"(?:\]\(|<img\s+src=\")([^)\":#]+)(?:#[^)\"]+)?(?:\)|\")")
