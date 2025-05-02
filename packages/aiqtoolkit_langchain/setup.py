from setuptools import setup


def build_dependencies():
    from setuptools_scm import get_version
    with open("requirements.in", "r", encoding="utf-8") as fh:
        requirements = fh.read()

    aiq_version = get_version('../..')
    requirements = requirements.format(version=aiq_version)

    return requirements.splitlines()


setup(install_requires=build_dependencies())
