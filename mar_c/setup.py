import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy.distutils.system_info import get_info

ext_modules = [
    Extension("hamilton_filter", ["hamilton_filter.pyx"])
]

setup(
    name = "Hamilton Filter",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)