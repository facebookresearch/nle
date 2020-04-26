#!/usr/bin/env python

import sysconfig
import shutil
import os
import sys
import subprocess
import setuptools
import setuptools.command.build_ext
import setuptools.command.develop


def make_install_nethack():
    # TODO(NN): we should make this an extension and build into the wheel,
    #           then we can use build_ext.spawn rather than subprocess.
    #           This would also allow pip / setup.py uninstall to remove nethack
    #           cleanly. See #44.
    if sys.platform == "darwin":
        setupcmd = ["sh", "setup.sh", "hints/macosx"]
    else:
        setupcmd = ["sh", "setup.sh", "hints/linux"]

    if subprocess.call(setupcmd, cwd="sys/unix") != 0:
        sys.exit(-1)

    build_env = os.environ.copy()

    if "PREFIX" not in build_env:
        build_env["PREFIX"] = sysconfig.get_config_var("base")

    cpath = build_env.get("CPATH", "").split(":")
    cpath.append(os.path.join(build_env["PREFIX"], "include"))
    build_env["CPATH"] = ":".join(cpath)

    if subprocess.call("make", env=build_env) != 0:
        sys.exit(-1)

    if subprocess.call(["make", "install"], env=build_env) != 0:
        sys.exit(-1)


def copy_fb_interface():
    source_path = "win/rl/nle/fbs"
    target_path = "nle/fbs"
    if os.path.exists(target_path):
        print(f"Found existing {target_path}-- Removing old package.")
        shutil.rmtree(target_path)
    print(f"Copying {source_path} to {target_path}")
    shutil.copytree(source_path, target_path)


class Develop(setuptools.command.develop.develop):
    """Customized setuptools develop command."""

    def run(self):
        # We need to copy the compiled python fb interface, otherwise we won't
        # be able to import the respective module when using the local package.
        self.execute(copy_fb_interface, [], msg="Copying python flatbuffer interface")
        super().run()


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


cwd = os.path.dirname(os.path.abspath(__file__))
version = open("version.txt", "r").read().strip()
sha = "Unknown"

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
    version += "+" + sha[:7]
except FileNotFoundError:
    pass

version_path = os.path.join(cwd, "nle", "version.py")
with open(version_path, "w") as f:
    f.write("__version__ = '{}'\n".format(version))
    f.write("git_version = {}\n".format(repr(sha)))

if __name__ == "__main__":
    # Ideally we would do this by monkeypatching setuptools.setup, however this
    # is good enough for now.
    make_install_nethack()
    setuptools.setup(
        name="nle",
        version=version,
        description="The NetHack Learning Environment",
        author="Heinrich Kuttler",
        author_email="hnr@fb.com",
        entry_points={
            "console_scripts": [
                "nle-play = nle.scripts.play:main",
                "nle-ttyrec = nle.scripts.ttyrec:main",
                "nle-ttyplay = nle.scripts.ttyplay:main",
            ]
        },
        packages=setuptools.find_packages() + ["nle.fbs"],
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExt, "develop": Develop},
        setup_requires=["pybind11>=2.2"],
        install_requires=[
            "pybind11>=2.2",
            "numpy>=1.16",
            "gym>=0.15",
            "pyzmq>=19.0.0",
            "flatbuffers>=1.10",
        ],
        extras_require={
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
        },
        package_dir={"nle.fbs": "win/rl/nle/fbs"},
    )
