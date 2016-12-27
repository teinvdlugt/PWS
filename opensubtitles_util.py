from __future__ import print_function

import gzip
import os
import tarfile
import urllib

dialogue_url = "http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/en.raw.tar.gz"
dialogue_file = "bigfile.txt"
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
        print("This will take very, very long...")
        file_path, _ = urllib.urlretrieve(url, file_path)
        file_info = os.stat(file_path)
        print("Successfully downloaded", filename, file_info.st_size, "bytes")
    else:
        print("File was already downloaded")
    return file_path


def maybe_unzip(fname, new_path):
    if os.path.exists(new_path):
        print("File already extracted")
        return new_path

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


def get_all_files_recursively(dir, ext=""):
    """Get all files in the directory recursively, that end with ext (optional)"""
    result = []
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        if not os.path.isfile(file_path):
            result = result + get_all_files_recursively(file_path, ext=ext)
        elif file.endswith(ext):
            result.append(file_path)
            break  # It seems that each .xml.gz file in one directory has the same subtitles
    return result


def process_zipped_xml_file(file_path):
    result = ""
    with gzip.open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith('<') or len(line) == 0 \
                    or line.startswith('[') or line.endswith(':') or line.startswith('('):
                # '<' often indicates an xml tag and '[', '(' and ':' a hearing-impaired thing
                continue
            else:
                if line.startswith('-'):
                    line = line[1:]
                result += line.strip().lower() + "\n"
    return result


def process_data(data_dir):
    zipped_file = maybe_download(data_dir, "opensub2016.tar.gz", dialogue_url)
    unzipped_path = maybe_unzip(zipped_file, os.path.join(data_dir, unzipped))
    files = get_all_files_recursively(unzipped_path, "xml.gz")
    print("Processing files to dialogue file...")
    with open(dialogue_file, 'w') as f:
        for file_path in files:
            f.write(process_zipped_xml_file(file_path))


def split_data(orig_file, train_file, test_file, percentage=10):
    """
Split the data from the original file into a train file and test file.
    :param orig_file: The original file to split.
    :param train_file: The path to store the train data
    :param test_file: The path to store the test data
    :param percentage: The percentage of the original file to go into the test file.
                       The rest will go into the train file.
    """
    # Count number of lines in orig_file
    num_lines = 0
    try:
        with open(orig_file) as f:
            for i, l in enumerate(f):
                pass
            num_lines = i + 1
    except:  # Just to be sure
        print("Splitting file wasn't successful")

    test_data_lines = int(num_lines / 100. * percentage)
    with open(orig_file) as orig_f:
        current_line = 0
        # Write test data
        with open(test_file, 'w') as test_f:
            while current_line < test_data_lines:
                test_f.write(orig_f.readline())
                current_line += 1
        # Write train data
        with open(train_file, 'w') as train_f:
            while current_line < num_lines:
                train_f.write(orig_f.readline())
                current_line += 1


def get_data(data_dir):
    if not (os.path.exists(os.path.join(data_dir, train_file))
                    and os.path.exists(os.path.join(data_dir, test_file))):
        # Download data, remove XML tags and put into one big file (dialogue_file)
        if not os.path.exists(os.path.join(data_dir, dialogue_file)):
            process_data(data_dir)
        # Split data into train_file and test_file
        print("Splitting data into train and test data...")
        split_data(os.path.join(data_dir, dialogue_file), os.path.join(data_dir, train_file),
                   os.path.join(data_dir, test_file))
        print("Done!")
    else:
        print("The data was already downloaded and processed")
    return os.path.join(data_dir, train_file), os.path.join(data_dir, test_file)
