import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="pyusadel",
    version="0.0.1",
    author="Andrea Maiani",
    author_email="andrea.maiani@nbi.ku.dk",
    packages=["pyusadel"],
    description="Usadel equation solver",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/maiani/pyusadel",
    license="MIT",
    python_requires=">=3.9",
    install_requires=["numpy>=1.21", "scipy>=1.7"],
)
