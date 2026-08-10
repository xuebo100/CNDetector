"""Builds the native extensions for PyPDMS."""

import argparse
import pathlib
import shutil
import sys
from subprocess import check_call


def parse_args():
    parser = argparse.ArgumentParser(prog="build_extensions")
    parser.add_argument("--build_dir", default="build")
    parser.add_argument("--build_type", default="release",
                        choices=["debug", "debugoptimized", "release"])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--additional", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def clean(build_dir: pathlib.Path, install_dir: pathlib.Path):
    if build_dir.exists():
        shutil.rmtree(build_dir)
    for extension in install_dir.rglob("*.so"):
        extension.unlink()
    for extension in install_dir.rglob("*.pyd"):
        extension.unlink()


def configure(build_dir: pathlib.Path, build_type: str, *additional: list[str]):
    cwd = pathlib.Path.cwd()
    is_windows = sys.platform.startswith("win")
    args = [
        build_dir,
        "--buildtype", build_type,
        f"-Dpython.platlibdir={cwd.absolute()}",
        f"-Dstrip={'true' if build_type == 'release' else 'false'}",
        f"-Db_coverage={'true' if build_type != 'release' else 'false'}",
        *(("-Db_vscrt=mt",) if is_windows else ()),
        *additional,
    ]
    cmd = "configure" if build_dir.exists() else "setup"
    check_call(["meson", cmd, *args])


def compile(build_dir: pathlib.Path, verbose: bool):
    args = ["-C", build_dir] + (["--verbose"] if verbose else [])
    check_call(["meson", "compile", *args])


def install(build_dir: pathlib.Path):
    check_call(["meson", "install", "-C", build_dir])


def build(build_dir: pathlib.Path, build_type: str, verbose: bool, *additional: list[str]):
    configure(build_dir, build_type, *additional)
    compile(build_dir, verbose)
    install(build_dir)


def main():
    args = parse_args()
    cwd = pathlib.Path.cwd()
    build_dir = cwd / args.build_dir

    if args.clean:
        install_dir = cwd / "pypdms"
        clean(build_dir, install_dir)

    build(build_dir, args.build_type, args.verbose, *args.additional)


if __name__ == "__main__":
    main()
