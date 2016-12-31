"""This file is used by Google Cloud ML software to know which python packages
to package and upload to the Google servers. If it doesn't exist, and the command
gcloud beta ml jobs submit training $JOB_NAME --package-path='chatbot' --module-name='chatbot.main'
is executed, the other packages, e.g. embedded_chars, won't be packaged and uploaded and
ImportErrors will be thrown.
"""

from setuptools import setup

# TODO When chatbot.word module is ready for Cloud ML, it should be added to the packages list.

if __name__ == '__main__':
    setup(name='chatbot', packages=['chatbot', 'chatbot.embedded_chars', 'chatbot.data_utils'])
