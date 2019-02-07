from setuptools import setup,find_packages
import versioneer

requires = [
    'numpy>=1.15.1',
    'scipy>=1.0.1',
    'pandas>=0.23.0',
    'scikit-learn>=0.20.1',
    'matplotlib',
    'seaborn>=0.9.0',
    'statsmodels',
    'fuzzywuzzy',
    'synapseclient',
]
VERSION = versioneer.get_version()
DESCRIPTION = ('A comprehensive collection of tools for analysing segmented Cycif data.')

setup(name='cycifsuite',
      version=VERSION,
      description=DESCRIPTION,
      url='https://github.com/yunguan-wang/cycif_analysis_suite',
      author='Yunguan Wang',
      author_email='yunguan_wang@hms.harvard.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=requires,
      zip_safe=False)
