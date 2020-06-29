#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# List of environment variables used:
#
#  NLE_PACKAGE_NAME
#    Prefix of the generated package (defaults to "nle").
#
#  NLE_BUILD_RELEASE
#    If set, builds wheel (s)dist such as to prepare it for upload to PyPI.
#
import os
import pathlib
import sys
import subprocess

import setuptools
import setuptools.command.build_ext


class CMakeBuild(setuptools.command.build_ext.build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        os.makedirs(self.build_temp, exist_ok=True)

        src_path = pathlib.Path(__file__).parent.resolve()
        # self.build_lib is also a good option, but it doesn't play nicely with
        # develop mode.
        out_path = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.resolve()

        cmake_cmd = [
            "cmake",
            src_path,
            "-DPYTHON_SRC_PARENT={}".format(out_path),
            # NOTE: This makes sure that cmake knows which python it is
            # compiling against.
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_INSTALL_PREFIX={}".format(sys.base_prefix),
        ]
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)
        subprocess.check_call(["make"], cwd=self.build_temp)
        subprocess.check_call(["make", "fbs"], cwd=self.build_temp)
        subprocess.check_call(["make", "install"], cwd=self.build_temp)


packages = [
    "nle",
    "nle.env",
    "nle.nethack",
    "nle.agent",
    "nle.scripts",
    "nle.tests",
    # NOTE: nle.fbs will be created at build time
]

entry_points = {
    "console_scripts": [
        "nle-play = nle.scripts.play:main",
        "nle-ttyrec = nle.scripts.ttyrec:main",
        "nle-ttyplay = nle.scripts.ttyplay:main",
    ]
}


extras_deps = {
    "dev": [
        "pre-commit>=2.0.1",
        "black>=19.10b0",
        "cmake_format>=0.6.10",
        "flake8>=3.7",
        "flake8-bugbear>=20.1",
        "pytest>=5.3",
        "sphinx>=2.4.4",
        "sphinx-rtd-theme==0.4.3",
    ],
    "agent": ["torch>=1.3.1"],
}

extras_deps["all"] = [item for group in extras_deps.values() for item in group]


if __name__ == "__main__":
    package_name = os.getenv("NLE_PACKAGE_NAME", "nle")
    cwd = os.path.dirname(os.path.abspath(__file__))
    version = open("version.txt", "r").read().strip()
    sha = "Unknown"

    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        pass

    if sha != "Unknown" and not os.getenv("NLE_RELEASE_BUILD"):
        version += "+" + sha[:7]
    print("Building wheel {}-{}".format(package_name, version))

    version_path = os.path.join(cwd, "nle", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

    with open("README.md") as f:
        long_description = f.read()

    setuptools.setup(
        name=package_name,
        version=version,
        description=(
            "The NetHack Learning Environment (NLE): "
            "a reinforcement learning environment based on NetHack"
        ),
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="The NLE Dev Team",
        url="https://github.com/facebookresearch/nle",
        license="NetHack General Public License",
        entry_points=entry_points,
        packages=packages,
        ext_modules=[setuptools.Extension("nlehack", sources=[])],
        cmdclass={"build_ext": CMakeBuild},
        setup_requires=["pybind11>=2.2"],
        install_requires=[
            "pybind11>=2.2",
            "numpy>=1.16",
            "gym>=0.15",
            "pyzmq>=19.0.0",
            "flatbuffers>=1.10",
        ],
        extras_require=extras_deps,
        python_requires=">=3.5",
        classifiers=[
            "License :: OSI Approved :: Nethack General Public License",
            "Development Status :: 4 - Beta",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: C",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Games/Entertainment",
        ],
    )
