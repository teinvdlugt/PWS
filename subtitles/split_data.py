"""
Run this script with parameters:
    1) the original dialogue file
    2) the filename for train data
    3) the filename for test data
    4) optional: the percentage of the original data to put in the test data, defaults to 10
"""

import sys

orig_file = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]
percentage = 10
if len(sys.argv) > 4:
    percentage = sys.argv[4]

lines = []
with open(orig_file, 'r') as f:
    for line in f:
        lines.append(line)

train_data_lines = len(lines) / 100 * (100 - percentage)

current_line = 0
with open(train_file, 'w') as f:
    while current_line < train_data_lines:
        f.write(lines[current_line])
        current_line += 1

with open(test_file, 'w') as f:
    while current_line < len(lines):
        f.write(lines[current_line])
        current_line += 1
