import os
import re

from setuptools import find_packages, setup


_deps = [
    "torch",
    "xformers",
    "diffusers==0.27.0",
    "transformers",
    "accelerate",
    "fire",
    "omegaconf",
    "cuda-python",
    "onnx==1.15.0",
    "onnxruntime==1.16.3",
    "protobuf==3.20.2",
    "colored",
    "einops",
    # "pywin32;sys_platform == 'win32'"
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["xformers"] = deps_list("xformers")
extras["torch"] = deps_list("torch", "accelerate")
extras["tensorrt"] = deps_list("protobuf", "cuda-python", "onnx", "onnxruntime", "colored")

extras["dev"] = extras["xformers"] + extras["torch"] + extras["tensorrt"]

install_requires = [
    deps["fire"],
    deps["omegaconf"],
    deps["diffusers"],
    deps["transformers"],
    deps["accelerate"],
]

setup(
    name="streamv2v",
    version="0.1.0",
    description="real-time video-to-video translation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion pytorch stable diffusion audioldm streamdiffusion real-time",
    license="UT Austin Research LICENSE",
    author="Feng Liang, Akio Kodaira, Chenfeng Xu, Masayoshi Tomizuka, Kurt Keutzer, Diana Marculescu",
    author_email="jeffliang@utexas.com",
    url="",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"streamv2v": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    extras_require=extras,
)
