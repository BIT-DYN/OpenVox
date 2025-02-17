from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# print(os.path.dirname(os.path.abspath(__file__)))

setup(
    name="tsdf_cuda",
    packages=['tsdf_cuda'],
    ext_modules=[
        CUDAExtension(
            name="integrate_kernel_cuda",
            sources=[
            "integrate_kernel.cu",
            "ext.cpp"],)
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
