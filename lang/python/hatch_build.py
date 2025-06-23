import os
import platform
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from sysconfig import get_platform
from tempfile import gettempdir
from zipfile import ZipFile

from auditwheel.wheel_abi import analyze_wheel_abi
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PARENT_DIR = Path(__file__).parent
LIBMVSR_SOURCE_DIR = PARENT_DIR.parents[1] / "mvsr"
SDIST_SOURCE_DIR = PARENT_DIR / "libmvsr"
TARGET_DIR = PARENT_DIR / "mvsr" / "lib"

LIBRARY_EXTENSIONS = ["so", "dylib", "dll"]


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if not os.environ.get("CI"):
            print(f"building '{LIBMVSR_SOURCE_DIR.name}' library ...")
            self.build_library()

        build_data["tag"] = f"py3-none-{self.get_platform_tag()}"

    def build_library(self):
        is_sdist_build = SDIST_SOURCE_DIR.is_dir()
        source_dir = SDIST_SOURCE_DIR if is_sdist_build else LIBMVSR_SOURCE_DIR

        TARGET_DIR.mkdir(exist_ok=True)

        if shutil.which("nix") and not is_sdist_build:
            subprocess.run(["nix", "build"], cwd=source_dir)
            out_dir = source_dir / "result" / "lib"
        else:
            out_dir = source_dir / "build"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir()

            generator = (
                ["-G", "Visual Studio 17 2022"]
                if platform.system() == "Windows"
                else []
            )
            subprocess.run(
                ["cmake", "-B", out_dir.name, "-DTESTING=off", *generator],
                cwd=source_dir,
            )
            subprocess.run(
                ["cmake", "--build", out_dir.name, "--config", "Release"],
                cwd=source_dir,
            )

        for path in chain(*(out_dir.glob(f"*.{ext}") for ext in LIBRARY_EXTENSIONS)):
            if (target_path := TARGET_DIR / path.name).is_file():
                target_path.unlink()
            shutil.copy(path, target_path)

    def get_platform_tag(self):
        platform_tag = get_platform().replace("-", "_").replace(".", "_")
        print("platform_tag 1", platform_tag)

        if platform.system() == "Linux":
            dummy_wheel_path = Path(gettempdir()) / "dummy.whl"
            with ZipFile(dummy_wheel_path, "w", strict_timestamps=False) as zip_file:
                records_str = ""
                for lib_path in chain(
                    *(TARGET_DIR.glob(f"*.{ext}") for ext in LIBRARY_EXTENSIONS)
                ):
                    zip_file.write(lib_path, lib_path.name)
                    records_str += f"{lib_path.name},,\n"
                zip_file.writestr("dummy.dist-info/RECORD", records_str)
            wheel_info = analyze_wheel_abi(
                None, None, dummy_wheel_path, frozenset(), False, False
            )

            if wheel_info.external_refs[wheel_info.policies.lowest.name].libs:
                platform_tag = f"{wheel_info.overall_policy.name}_{platform_tag}"

        elif platform.system() == "Darwin":
            product_version = "_".join(
                subprocess.check_output(["sw_vers -productVersion"])
                .decode()
                .split(".")[:2]
            )
            platform_tag = f"macosx_{product_version}_{platform.uname().machine}"

        print("platform_tag 2", platform_tag)

        return platform_tag
