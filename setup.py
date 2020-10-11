from setuptools import setup
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'pypi-description.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='condense',
    packages=['condense'],
    version='0.2.0',
    license='MIT',
    description='Neural Network Pruning Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lucas Sas Brunschier',
    author_email='lucassas@live.de',
    url='https://github.com/SirBubbls/condense',
    download_url='https://github.com/SirBubbls/condense/archive/v_01.tar.gz',
    keywords=['pruning', 'ai', 'machine learning', 'tensorflow', 'framework'],
    python_requires='>=3.8',
    install_requires=[
        'keras',
        'tensorflow',
        'numpy'
    ],
    tests_require=[
        'pillow',
        'tensorflow_datasets'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
