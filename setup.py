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
#  NLE_BUILD_FASTER
#    If set, runs `make -j` instead of `make`, when building NetHack.
#    NOTE: we don't make use of MAKEFLAGS (which would be The Right Thing To
#          Do™) because `make install` is not robust against it on all systems,
#          and this is an easier patch towards enabling less annoying
#          development cycles.
import sysconfig
import shutil
import os
import sys
import subprocess
import setuptools
import setuptools.command.build_ext
import setuptools.command.develop
import setuptools.command.install
import distutils.command.build


def make_install_nethack():
    if sys.platform == "darwin":
        hints = "hints/macosx-nle"
    else:
        hints = "hints/linux-nle"
    setupcmd = ["sh", "setup.sh", hints]

    if subprocess.call(setupcmd, cwd="sys/unix") != 0:
        print("Failed to generate NetHack makefiles with {}".format(hints))
        sys.exit(-1)

    print("Generated NetHack makefiles with {}.".format(hints))

    build_env = os.environ.copy()

    if "PREFIX" not in build_env:
        build_env["PREFIX"] = sysconfig.get_config_var("base")

    cpath = build_env.get("CPATH", "").split(":")
    cpath.append(os.path.join(build_env["PREFIX"], "include"))
    build_env["CPATH"] = ":".join(cpath)

    makecmd = ["make"]
    if os.getenv("NLE_BUILD_FASTER") is not None:
        makecmd += ["-j"]

    if subprocess.call(makecmd, env=build_env) != 0:
        print("Failed to build NetHack")
        sys.exit(-1)
    print("Successfully built NetHack")

    if subprocess.call(["make", "install"], env=build_env) != 0:
        print("Failed to install NetHack")
        sys.exit(-1)
    print("Successfully installed NetHack")


def build_fbs():
    source_path = "win/rl"
    cmd = ["flatc", "-o", "win/rl", "--python", "win/rl/message.fbs"]
    err = subprocess.call(cmd)
    if err != 0:
        print("Error while building fbs python interface: {}".format(err))
        sys.exit(-1)
    print("Successfully built fbs files from {}".format(source_path))


def copy_fbs():
    source_path = "win/rl/nle/fbs"
    target_path = "nle/fbs"
    if os.path.exists(target_path):
        print("Found existing {} -- Removing old package".format(target_path))
        shutil.rmtree(target_path)
    print("Copying {} to {}".format(source_path, target_path))
    shutil.copytree(source_path, target_path)


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    setuptools.Extension(
        "nle.nethack.helper",
        ["win/rl/helper.cc"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            "include",
        ],
        language="c++",
        # Warning: This should stay in sync with the Makefiles, otherwise
        # off-by-one errors might happen. E.g. there's a MAIL deamon only
        # if the MAIL macro is set.
        extra_compile_args=["-DNOCLIPPING", "-DNOMAIL", "-DNOTPARMDECL"],
        # This requires `make`ing NetHack before.
        extra_link_args=["src/monst.o", "src/decl.o", "src/drawing.o"],
    )
]


class BuildExt(setuptools.command.build_ext.build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++14")
            opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args += opts
            if sys.platform == "darwin":
                ext.extra_link_args += ["-stdlib=libc++"]

        super().build_extensions()


class Build(distutils.command.build.build):
    """Customized distutils build command."""

    def run(self):
        self.execute(make_install_nethack, [], msg="Building and installing NetHack...")
        super().run()


class Install(setuptools.command.install.install):
    """Customized setuptools develop command."""

    def run(self):
        self.execute(make_install_nethack, [], msg="Building and installing NetHack...")
        super().run()


class Develop(setuptools.command.develop.develop):
    """Customized setuptools develop command."""

    def run(self):
        self.execute(make_install_nethack, [], msg="Building and installing NetHack...")
        super().run()


packages = [
    "nle",
    "nle.env",
    "nle.nethack",
    "nle.fbs",
    "nle.agent",
    "nle.scripts",
    "nle.tests",
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

    # nle.fbs is declared as package, and it is hard to make sure that we
    # override all behaviour of pip [build_py|install|develop|...], without
    # hitting corner cases. So we _always_ build the python fbs files and copy
    # them into nle/fbs. This does mean that effectively setup.py now depends
    # on`flatc`, however this is not a terrible constraint, as 95% of operations
    # already needed it.
    build_fbs()
    copy_fbs()

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
        ext_modules=ext_modules,
        cmdclass={
            "build": Build,
            "build_ext": BuildExt,
            "install": Install,
            "develop": Develop,
        },
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
