from setuptools import setup, find_packages

setup(
    name="jnanaverse",
    version="2.0.0",
    author="Dharmin Joshi / DevKay",
    description="Open-source multi-task NLP framework: generation, classification, similarity.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "tokenizers>=0.19.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "adapters": ["adapters>=0.2.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
