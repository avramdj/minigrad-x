import os
import subprocess
import sys
from glob import glob

from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["--", "-j2"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


def generate_stubs():
    from mypy.stubgen import main as stubgen_main

    print("Generating stubs…")
    for so_path in glob("minigradx/*.so"):
        name = os.path.splitext(os.path.basename(so_path))[0]
        real_mod = name.split(".", 1)[0]
        full_mod = f"minigradx.{real_mod}"
        print(f"Generating stubs for {full_mod}…")
        stubgen_main(["-m", full_mod, "-o", "."])


class GenerateStubs(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        generate_stubs()


def main():
    setup(
        name="minigradx",
        version="0.0.1",
        author="Avram Djordjevic",
        author_email="avramdjordjevic2@gmail.com",
        description="A small autograd engine",
        ext_modules=[CMakeExtension("minigradx._C")],
        cmdclass={
            "build_ext": CMakeBuild,
            "generate_stubs": GenerateStubs,
        },
        zip_safe=False,
        include_package_data=True,
        install_requires=["numpy"],
        extras_require={
            "dev": ["pytest", "pytest-cov", "mypy", "setuptools", "ipython"],
            "cuda": [],
        },
    )


if __name__ == "__main__":
    main()
