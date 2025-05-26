from setuptools import setup, find_packages

setup(
    name="portfolio_optimizer_shared",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pika>=1.2.0",
        "aio-pika>=7.2.0",
    ],
    description="Shared modules for Portfolio Optimizer microservices",
    author="Portfolio Optimizer Team",
) 