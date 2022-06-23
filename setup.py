from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Education',
    'Operating System :: POSIX :: Linux',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3'
]

setup(
    name='eds',
    version='0.1',
    description='EDS FOR Dr. M.B.A.',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',  
    author='',
    author_email='',
    license='GPLv3', 
    classifiers=classifiers,
    keywords='Disease', 
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas'
    ] 
)