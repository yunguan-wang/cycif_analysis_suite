from setuptools import setup

setup(name='cycifsuite',
      version='0.1',
      description='A comprehensive collection of tools for analysing segmented Cycif data.',
      url='https://github.com/yunguan-wang/cycif_analysis_suite',
      author='Yunguan Wang',
      author_email='yunguan_wang@hms.harvard.edu',
      license='MIT',
      packages=['cycifsuite'],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'scikit-learn',
          'matplotlib',
          'seaborn',
          'statsmodels',
          'fuzzywuzzy',
          'synapseclient'
      ],
      zip_safe=False)
