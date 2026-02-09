from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="high-dimensional-consumer-data-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade ETL framework for high-dimensional consumer behavior data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/High-Dimensional-Consumer-Data-Pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "mlops": [
            "mlflow>=1.30.0",
            "dvc>=2.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hdcdp-pipeline=example_pipeline:main",
        ],
    },
    keywords="etl data-engineering machine-learning quantitative-finance alternative-data xgboost lightgbm matrix-factorization time-series",
)
