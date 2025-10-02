from pathlib import Path

import nox

# pyright: basic

nox.options.default_venv_backend = "uv"

MIN_NUMPY_VERSION = {
    "3.10": "1.21.*",
    "3.11": "1.23.*",
    "3.12": "1.26.*",
    "3.13": "2.1.*",
}
PINNED_NUMPY_VERSION = "2.2.*"

COVERAGE_CONFIG = Path(__file__).parents[1] / "pyproject.toml"


@nox.session
def test_simple(session: nox.Session):
    session.install("matplotlib")
    session.install("pytest", "pytest-cov")
    session.install("..")
    session.run("pytest", "-v", "--no-cov", "-k", "simple")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
@nox.parametrize("numpy", ["min", "pinned", "latest"])
def test_versions(session: nox.Session, numpy: str):
    match numpy:
        case "min":
            assert isinstance(session.python, str)
            numpy_version = f"numpy=={MIN_NUMPY_VERSION[session.python]}"
        case "pinned":
            numpy_version = f"numpy=={PINNED_NUMPY_VERSION}"
        case "latest":
            numpy_version = "numpy"

    session.install("matplotlib")
    session.install(numpy_version, "--no-build")
    session.run("python", "-c", 'import numpy; print("numpy version:", numpy.version.version)')
    session.install("..")

    session.install("pytest", "pytest-cov")
    session.run("pytest", "-v", "--no-cov")


@nox.session
def coverage(session: nox.Session):
    session.install("matplotlib")
    session.install("pytest", "pytest-cov")
    session.install("..")
    session.run("pytest", "-v", "--cov-config", COVERAGE_CONFIG)
