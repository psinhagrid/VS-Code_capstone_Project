"""Setup script for PPE Detection System"""

from setuptools import setup, find_packages
from pathlib import Path

project_root = Path(__file__).parent
long_description = (project_root / "README.md").read_text() if (project_root / "README.md").exists() else ""

setup(
    name="ppe-detector",
    version="2.0.0",
    description="Personal Protective Equipment Violation Detection System using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ppe-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.2.0",
        "opencv-python>=4.9.0",
        "numpy>=1.26.0",
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "cvzone>=1.5.6",
        "pillow>=10.3.0",
        "scikit-image>=0.23.0",
        "filterpy>=1.4.5",
        "scipy>=1.13.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "kafka": ["confluent-kafka>=2.3.0"],
        "dev": [
            "pytest>=8.1.0",
            "black>=24.3.0",
            "flake8>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ppe-detector=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="computer-vision deep-learning yolo ppe safety detection",
)
