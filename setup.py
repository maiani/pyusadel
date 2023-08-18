from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    description = fh.read()

setup(
    name="pyusadel",
    version="0.2.0",
    author="Andrea Maiani",
    author_email="andreamaiani@gmail.com",
    packages=find_packages(),
    description="Usadel equation solver",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/maiani/pyusadel",
    license="MIT",
    python_requires=">=3.9",
    install_requires=["numpy>=1.21", "scipy>=1.7"],
    extras_require={"extra": ["numba>=0.56"]},
)
