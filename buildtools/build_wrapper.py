"""
Wrapper around the build_extensions script for poetry build backend.
"""

import pathlib
import tempfile

from build_extensions import build, clean


def main():
    cwd = pathlib.Path.cwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = pathlib.Path(tmpdir)
        install_dir = cwd / "cndetector"

        clean(build_dir, install_dir)
        build(build_dir, build_type="release", verbose=False)


if __name__ == "__main__":
    main()
