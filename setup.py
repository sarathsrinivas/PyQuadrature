from setuptools import setup

setup(name='PyQuadrature',
      version='0.1',
      description='Integration quadrature for spherical integrals.',
      url='http://github.com/sarathsrinivas/PyQuadrature.git',
      author='Sarath Srinivas S',
      author_email='srinix@pm.me',
      license='MIT',
      packages=['PyQuadrature'],
      install_requires=['torch', 'numpy', 'scipy'],
      zip_safe=False)
