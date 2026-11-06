from setuptools import find_packages, setup

setup(
    name="deep-research-agent",
    version="0.1.0",
    description="TRQA-Wiki2 Reproducibility & Deep Research Agent (BM25 + Search-R1)",
    packages=find_packages(where="."),
    python_requires=">=3.10",
)

