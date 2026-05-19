from setuptools import setup, find_packages

setup(
    name="doc-prompt-study",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5",
        "transformers>=4.50",
        "datasets",
        "Pillow",
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "PyYAML",
        "tqdm",
        "qwen-vl-utils",
        "bitsandbytes",
        "accelerate",
    ],
)
