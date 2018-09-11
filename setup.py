from setuptools import setup, find_packages

setup(
    name='dnovelARG',
    version='0.1',
    packages=find_packages(exclude=('database', 'model', 'logs', 'test')),
    include_package_data=True,
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
        dnovelARG=DeepNovelARG.entry:cli
    ''',
)
