from setuptools import find_packages, setup


package_name = "nn_bot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=("ab", "ab.*")),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy<2", "onnxruntime>=1.17"],
    zip_safe=True,
    maintainer="ABrain-One",
    maintainer_email="contact@abrain.one",
    description="ROS 2 image inference and ONNX benchmarking for robotic systems",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nn-bot=ab.bot.port:main",
            "nn-bot-pipeline=ab.bot.dataset_pipeline:main",
        ]
    },
)