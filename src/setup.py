from setuptools import setup, find_packages

setup(
    name='dnovelARG',
    version='0.1',
    packages=find_packages(),
    include_package_data=False,
    install_requires=[
        'Click',
        'BioPython',
        'ete3',
        'h5py',
        'tqdm',
        'tensorflow',
        'sklearn'
    ],
    entry_points='''
        [console_scripts]
        deepARG+=DeepNovelARG.entry:cli
    ''',
)
