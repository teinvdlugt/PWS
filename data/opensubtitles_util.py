from __future__ import print_function

import os
import tarfile
import urllib

dataset_url = "https://teinvdlugt.stackstorage.com/public-share/t8IBEatZsz4oUcN/preview?path=%2F&mode=full"
zipped_file = "osdataset.tar.gz"
train_file = "train.txt"
test_file = "test.txt"
unzipped = "unzipped/"


def maybe_download(directory, filename, url):
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        print("Downloading %s to %s..." % (url, file_path))
        print("This may take very, very long...")
        file_path, _ = urllib.urlretrieve(url, file_path)
        file_info = os.stat(file_path)
        print("Successfully downloaded", filename, file_info.st_size, "bytes")
    else:
        print("File was already downloaded")
    return file_path


def unzip(fname, new_path):
    print("Extracting %s to %s..." % (fname, new_path))
    tar = tarfile.open(fname, "r:gz")
    try:
        tar.extractall(path=new_path)
    except IOError:
        # Is always thrown at the end of the tar.gz file, it seems
        pass
    finally:
        tar.close()
        return new_path


def get_data(data_dir):
    if not (os.path.exists(os.path.join(data_dir, train_file)) and
            os.path.exists(os.path.join(data_dir, test_file))):
        # Download dataset
        downloaded_path = maybe_download(data_dir, zipped_file, dataset_url)
        # Extract dataset
        unzip(downloaded_path, data_dir)
        print("Done!")
    else:
        print("The data was already downloaded and processed")
    return os.path.join(data_dir, train_file), os.path.join(data_dir, test_file)
