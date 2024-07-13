from setuptools import setup, find_packages

setup(
    name='colab_research_tools',
    version='0.0.6',
    packages=find_packages(),
    install_requires=[
        'nltk==3.6.2',
        'fasttext==0.9.2',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'sentence-transformers==2.2.2',
        'python-Levenshtein',
        'transformers[torch]',
        'tqdm'
    ],
    author='Aleksandr Dzhumurat',
    author_email='adzhumurat@yandex.ru',
    description='Speed up Google Colab research',
    url='https://github.com/aleksandr-dzhumurat/colab_research_tools',
    license='MIT',
)
