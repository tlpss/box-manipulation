from setuptools import setup

setup(
    name="keypoint_detection",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    version="0.1",
    description="Pytorch Models, Modules etc for box-manipulation project",
    url="https://github.com/tlpss/box-manipulation",
    packages=["keypoint_detection"],
    # install_requires=[], # requirements are not handled by this package, since its use is mostly to provide easier use of imports.
)
