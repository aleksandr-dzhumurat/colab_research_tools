from setuptools import setup, find_packages

setup(
    name='colab_research_tools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'nltk==3.6.2',
        'fasttext==0.9.2',
    ],
    author='Aleksandr Dzhumurat',
    author_email='adzhumurat@yandex.cry',
    description='Speed up Google Colab research',
    url='https://github.com/aleksandr-dzhumurat/colab_research_tools',
    license='MIT',
)
