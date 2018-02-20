from setuptools import setup
from setuptools import find_packages


version = '0.1'

setup(name='keras-visualize-activations',
      version=version,
      description='Visualize keras layers and activations. Original author is Phillipe Rémy, found here: https://github.com/philipperemy',
      author='Toby Buckley',
      author_email='toby.buckley@offworld.ai',
      url='https://github.com/offworld-projects/keras-visualize-activations',
      license='MIT',
      install_requires=['keras', 'natsort', 'numpy', 'h5py'],
      packages=['keras_visualize'])
