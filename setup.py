"""This file is used by Google Cloud ML software to know which python packages
to package and upload to the Google servers. Add to the packages list the packages
that you want Cloud ML to upload to the Google servers.
"""

from setuptools import setup

if __name__ == '__main__':
    setup(name='chatbot', packages=['chatbot'], install_requires=['numpy', 'tensorflow', 'flask'])
