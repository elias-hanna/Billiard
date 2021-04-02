# Created by Giuseppe Paolo 
# Date: 21/12/18

from setuptools import setup, find_packages

setup(name='gym_billiard',
      install_requires=['gym', 'numpy', 'box2d-py', 'pygame'],
      version='1.0.0',
      packages=find_packages(),
      include_package_data=True,
      author="Giuseppe Paolo",
      author_email="giuseppe.paolo93@gmail.com",
      description="An OpenAI gym environment",
)
