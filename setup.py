from setuptools import setup, find_packages

setup(
    name="codewalk",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'codewalk=cw.cli:main',
        ],
    },
    install_requires=[
        'pydantic', 'litellm', 'rich', 'python-dotenv',
        'anthropic', 'openai', 'groq', 'matplotlib',
        'seaborn', 'langfuse'
    ],
)
