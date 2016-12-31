from __future__ import print_function

import os
import tarfile
import urllib

from tensorflow.python.platform import gfile

dataset_url = "https://teinvdlugt.stackstorage.com/public-share/t8IBEatZsz4oUcN/preview?path=%2F&mode=full"
zipped_file = "osdataset.tar.gz"
train_file = "train.txt"
test_file = "test.txt"
tokenized_dataset_url = "https://teinvdlugt.stackstorage.com/public-share/6fHjGRTHLIHfMWx/preview?path=%2F&mode=full"
zipped_tokenized_file = "OSDataset60M60vCh.tar.gz"


def maybe_download(directory, filename, url):
    if not gfile.Exists(directory):
        print("Creating directory %s" % directory)
        gfile.MkDir(directory)
    file_path = os.path.join(directory, filename)
    if not gfile.Exists(file_path):
        print("Downloading %s to %s..." % (url, file_path))
        print("This may take very, very long...")
        file_path, _ = urllib.urlretrieve(url, file_path)  # TODO This probably doesn't work with GCS buckets.
        # It should be avoided to use the above function, because I think it won't work with Google Cloud
        # Storage buckets, just as the normal Python open(file, mode) function doesn't work with the buckets.
        # But I don't know how to make any download function work with GCS, so until now I have uploaded
        # every necessary file to the bucket manually so nothing has to be downloaded.
        file_info = gfile.Stat(file_path)
        print("Successfully downloaded", filename, file_info.st_size, "bytes")
    else:
        print("File was already downloaded")
    return file_path


def unzip(fname, new_path):
    print("Extracting %s to %s..." % (fname, new_path))
    tar = tarfile.open(mode="r:gz", fileobj=gfile.Open(fname))
    try:
        tar.extractall(path=new_path)  # TODO Probably doesn't work with GCS buckets either...
    except IOError:
        # Is always thrown at the end of the tar.gz file, it seems
        pass
    finally:
        tar.close()
        return new_path


def get_data(data_dir):
    if not (gfile.Exists(os.path.join(data_dir, train_file)) and
            gfile.Exists(os.path.join(data_dir, test_file))):
        # Download dataset
        downloaded_path = maybe_download(data_dir, zipped_file, dataset_url)
        # Extract dataset
        unzip(downloaded_path, data_dir)
        print("Done!")
    else:
        print("The data was already downloaded and processed")
    return os.path.join(data_dir, train_file), os.path.join(data_dir, test_file)


def get_encoded_data(data_dir):
    print("Downloading tokenized data and vocabulary")
    if not (gfile.Exists(os.path.join(data_dir, "chars_test_ids60")) and
            gfile.Exists(os.path.join(data_dir, "chars_train_ids60")) and
            gfile.Exists(os.path.join(data_dir, "chars_vocab60"))):
        downloaded_file = maybe_download(data_dir, zipped_tokenized_file, tokenized_dataset_url)
        unzip(downloaded_file, data_dir)
    return (os.path.join(data_dir, "chars_train_ids60"),
            os.path.join(data_dir, "chars_test_ids60"),
            os.path.join(data_dir, "chars_vocab60"))
