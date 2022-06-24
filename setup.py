import sys, setuptools

if sys.version_info[0] < 3:
    sys.stdout.write("hdlib requires Python 3 or higher. Your Python your current Python version is {}.{}.{}"
                     .format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))

setuptools.setup(name='hdlib',
                 version='0.1.0',
                 author='Fabio Cumbo',
                 author_email='fabio.cumbo@gmail.com',
                 url='http://github.com/cumbof/hdlib',
                 license='LICENSE',
                 packages=setuptools.find_packages(),
                 description='Hyperdimensional Computing Library for building Vector Symbolic Architectures in Python',
                 long_description=open('README.md').read(),
                 long_description_content_type='text/markdown',
                 install_requires=[
                     "numpy"
                 ],
                 zip_safe=False)
