import os
import re
import shutil
import subprocess
import setuptools
import setuptools.command.build_py
from setuptools import find_packages, setup


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        try:
            config_path = os.path.join("subsync", "config.py")
            config_path_template = os.path.join("subsync", "config.py.template")
            if not os.path.isfile(config_path) and os.path.isfile(config_path_template):
                shutil.copyfile(config_path_template, config_path)
        except Exception:
            pass
        super().run()


def read_version():
    try:
        version_long = (
            subprocess.check_output(["git", "describe", "--tags"], stderr=subprocess.DEVNULL)
            .decode("UTF-8")
            .strip()
        )
        v = version_long[re.search(r"\d", version_long).start() :].split("-")
        version = "{}.{}".format(v[0], v[1]) if len(v) > 1 else "{}.0".format(v[0])
    except Exception:
        version_long = "2.0.0"
        version = "2.0.0"
    return version_long, version


setup(
    name="subsync",
    version=read_version()[1],
    author="Julien Chabrier (based on original SubSync by Michał Szymaniak)",
    url="https://github.com/Julienchh/subsync",
    description="Hybrid AI + Mathematical Subtitle Speech Synchronizer & Series Pipeline",
    license="GPLv3",
    python_requires=">=3.8",
    packages=find_packages(include=["subsync", "subsync.*"]),
    entry_points={
        "console_scripts": [
            "subsync=subsync.__main__:subsync",
            "subsync-cmd=subsync.__main__:subsync",
        ],
    },
    install_requires=[
        "certifi",
        "PyYAML",
        "requests>=2.0",
        "numpy>=1.22",
    ],
    extras_require={
        "ai": ["onnxruntime>=1.16", "faster-whisper>=1.0"],
        "GUI": ["wxPython>=4.0"],
        "test": ["pytest>=7.0"],
    },
    scripts=["bin/subsync", "bin/subsync-gui"],
    cmdclass={"build_py": build_py},
    package_data={
        "subsync": [
            "key.pub",
            os.path.join("img", "*.png"),
            os.path.join("img", "*.ico"),
            os.path.join("locale", "*", "LC_MESSAGES", "*.mo"),
        ],
    },
)
