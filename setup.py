"""Setup the package with pip."""


import os
import setuptools


# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


INSTALL_REQUIRES = [
    'jax>=0.2.21'
]


def _get_version() -> str:
  """Returns the package version.
  Adapted from:
  https://github.com/deepmind/dm-haiku/blob/d4807e77b0b03c41467e24a247bed9d1897d336c/setup.py#L22
  Returns:
    Version number.
  """
  path = 'fast_finite_width_ntk/__init__.py'
  version = '__version__'
  with open(path) as fp:
    for line in fp:
      if line.startswith(version):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g[version]  # pytype: disable=key-error
    raise ValueError(f'`{version}` not defined in `{path}`.')


setuptools.setup(
    name='fast-finite-width-ntk',
    version=_get_version(),
    license='Apache 2.0',
    author='Anonymous',
    author_email='iclr2022anon@gmail.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/iclr2022anon/fast_finite_width_ntk',
    download_url='https://github.com/iclr2022anon/fast_finite_width_ntk',
    project_urls={
        'Source Code': 'https://github.com/iclr2022anon/fast_finite_width_ntk',
        'Paper': 'https://openreview.net/forum?id=zLb9oSWy933',
    },
    packages=setuptools.find_packages(exclude=('presentation',)),
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Fast Finite Width Neural Tangent Kernel',
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Development Status :: 4 - Beta',
    ])
