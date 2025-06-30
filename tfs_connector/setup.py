import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='irym_tfs_connector',
    version='0.8.5',
    author='mahavoid',
    description='A small library that helps feed TensorFlow Serving REST API images.',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url='http://192.168.1.59:16326/simple/irym_tfs_connector',
    license='proprietary and confidential',
    packages=['irym_tfs_connector'],
    install_requires=['cv2',
                      'datetime',
                      'functools',
                      'inspect',
                      'itertools',
                      'logging',
                      'multiprocessing',
                      'numpy',
                      'orjson'
                      'os',
                      'requests',
                      'typing',
                      'scipy',
                      'dill',
                      'multiprocess']
)
