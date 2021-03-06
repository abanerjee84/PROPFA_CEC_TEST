from setuptools import Extension
from distutils.core import setup
from setuptools.command.test import test as TestCommand
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys

# cython = Extension('ea.cbenchmarks',
#    sources = ['ea/cbenchmarks.pyx'],
##    include_dirs = ['include/']
#)
sourcefiles = ['cec2005real/cec2005.pyx']

sourcefiles += ['cec2005real/api_cec2005.cc']

cec2005real = Extension("cec2005real.cec2005",
                        sourcefiles,
                        language="c++",
                        extra_compile_args=["-std=c++0x"],
                        libraries=["m"])  # Unix-like specific


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='cec2005real',
    version='0.1',
    author='Daniel Molina',
    author_email='daniel.molina@uca.es',
    maintainer='Daniel Molina',
    description='Package for benchmark for the Real \
    Optimization session on IEEE \
    Congress on Evolutionary Computation CEC\'2005',
    long_description=open('README.rst').read(),
    license='GPL V3',
    url='https://github.com/dmolina/cec2005real',
    packages=['cec2005real'],
    install_requires=['cython', 'numpy'],
    ext_modules=cythonize(cec2005real),
    package_data={'cec2005real': ['cdatafiles/*.txt']},
    tests_require=['pytest'],
    cmdclass={'build_ext': build_ext, 'test': PyTest},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    ]
)
