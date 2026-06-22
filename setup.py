from setuptools import setup, find_packages

setup(
    name='colab_research_tools',
    version='0.0.15',
    packages=find_packages(),
    install_requires=[
        'nltk==3.6.2',
        'pandas',
        'numpy',
        'python-Levenshtein',
        'tqdm',
        'google-api-python-client',
        'gspread'
    ],
    author='Aleksandr Dzhumurat',
    author_email='adzhumurat@yandex.ru',
    description='Speed up Google Colab research',
    url='https://github.com/aleksandr-dzhumurat/colab_research_tools',
    license='MIT',
)
