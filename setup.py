from glob import glob

from setuptools import setup

setup(
    name="rammp_feeding",
    version="0.2.3",
    packages=["rammp_feeding"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/rammp_feeding"]),
        ("share/rammp_feeding", ["package.xml"]),
        ("share/rammp_feeding/launch", glob("launch/*.py")),
        ("share/rammp_feeding/urdf", glob("urdf/*")),
        ("share/rammp_feeding/config", glob("config/*.yaml")),
    ] + [("share/rammp_feeding/config/" + d, glob("config/%s/*.yaml" % d)) for d in ("bowl", "camera", "food", "utensil", "machines")],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["rammp = rammp_feeding.cli:main"]},
)
