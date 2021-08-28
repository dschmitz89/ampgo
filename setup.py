from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == '__main__':
    
    setup(
        name='ampgo',   
        version='1.0', 
        description='Adaptive Memory Programming for Global Optimization',
        url='https://github.com/dschmitz89/ampgo/',
        #download_url='https://github.com/dschmitz89/simplenlopt/archive/refs/tags/1.0.tar.gz',
        author='Daniel Schmitz',
        license='MIT',
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=['ampgo'],
        install_requires=[
            'numpy',
            'scipy>0.11',
        ]            
    )