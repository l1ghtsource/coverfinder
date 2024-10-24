from setuptools import setup, find_packages

setup(
    name='cover_finder',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchaudio>=0.9.0',
        'numpy>=1.19.2',
        'pandas>=1.2.3',
        'wandb>=0.12.0',
        'tqdm>=4.62.3',
        'PyYAML>=5.4.1',
    ],
    author='Lightsource',
    author_email='oiicavvaciio@gmail.com',
    description='A deep learning model for finding cover songs',
    keywords='music, deep learning, cover songs',
    url='https://github.com/l1ghtsource/coverfinder',
)
