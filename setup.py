from setuptools import setup

install_requires = [
    'pandas>=0.25.0',
    'numpy>=1.15.4',
    'functools'
]


setup(name='src',
      version='0.0.1',
      description='Produced for for EL technical assessment',
      author='Steve Betts',
      author_email='stevo.betts@gmail.com',
      install_requires=install_requires,
      packages=['src'])
