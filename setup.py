from setuptools import setup, find_packages

setup(
    name="quantum_market_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0"
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    }
)