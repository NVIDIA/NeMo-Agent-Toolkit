from datetime import datetime

from setuptools import setup


def build_dependencies():
    with open("/ci_tmp/build.log", "a", encoding="utf-8") as log:
        log.write(f"Building dependencies...{datetime.now()}\n")

    from setuptools_scm import get_version
    with open("requirements.in", "r", encoding="utf-8") as fh:
        requirements = fh.read()

    aiq_version = get_version('../..')
    requirements = requirements.format(version=aiq_version)

    return requirements.splitlines()


setup(install_requires=build_dependencies())
