from distutils.core import setup

setup(
    name='condense',
    packages=['condense'],
    version='0.01-beta',
    license='MIT',
    description='Neural Network Pruning Framework',
    author='Lucas Sas Brunschier',
    author_email='lucassas@live.de',
    url='https://github.com/SirBubbls/condense',
    download_url='https://github.com/SirBubbls/condense/archive/v_01.tar.gz',
    keywords=['pruning', 'ai', 'machine learning', 'tensorflow', 'framework'],
    install_requires=[
        'keras',
        'tensorflow',
        'numpy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
