file_path = "./os2/OpenSubtitles2016.raw.en"  # Contains a very large file (10GB) with 337M sentences
train_path = "./os2/train.txt"
test_path = "./os2/test.txt"
train_data_lines = 60000000  # Number of sentences to put in training data
test_data_lines = 5000000  # Number of sentences to put in test data


def clean_line(string):
    if string.startswith(('-', '\"', '\'')):
        string = string[1:].strip()
    if string.endswith(('\"', '\'')):
        string = string[:-1].strip()

    # Hearing impaired stuff
    if string.startswith('['):
        try:
            string = string[string.index(']') + 1:].strip()
        except ValueError:
            return None

    if string.startswith('('):
        try:
            string = string[string.index(')') + 1:].strip()
        except ValueError:
            return None

    if string.startswith('#'):
        # Symbol is often used when a song is subtitled
        return None

    return string


current_line = 0
with open(file_path) as f:
    print("Writing train data...")
    with open(train_path, 'w') as train_f:
        while current_line < train_data_lines:
            line = clean_line(f.readline().strip())
            if line:
                train_f.write(line + "\n")
                current_line += 1
            if current_line % 100000 == 0:
                print("Line %d" % current_line)
    print("Writing test data...")
    with open(test_path, 'w') as test_f:
        while current_line < train_data_lines + test_data_lines:
            line = clean_line(f.readline().strip())
            if line:
                test_f.write(line + "\n")
                current_line += 1
            if current_line % 100000 == 0:
                print("Line %d" % (current_line - train_data_lines))
