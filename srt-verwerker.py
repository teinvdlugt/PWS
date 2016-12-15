import sys
import os

def lineIsNumber(line):
    try:
        test = int(line.strip())
        return True
    except:
        return False


newfilename = sys.argv[1]
filenames = sys.argv[2:]

# Check if filenames contains any directories
filesindirs = []
for filename in filenames:
    if os.path.isdir(filename):
        filenames.remove(filename)
        files = os.listdir(filename)
        for f in files:
            if (os.path.isfile(os.path.join(filename, f)) and f.endswith(".srt")):
                filesindirs.append(os.path.join(filename, f))
filenames = filenames + filesindirs

if len(filenames) == 0:
    print "Provide at least two arguments"

newfile = open(newfilename, 'w')
for filename in filenames:
    print "Processing file %s" % filename

    with open(filename, 'r') as f:
        count = 0
        prevLineWasUtt = False
        for line in f:
            count += 1

            if not lineIsNumber(line) and line.count("-->") == 0 and len(line.strip()) > 0:
                if line.startswith("-"):
                    line = line[1:]
                if prevLineWasUtt:
                    newfile.write(" ")
                else:
                    newfile.write("\n")
                newfile.write(line.strip())
                prevLineWasUtt = True
            else:
                prevLineWasUtt = False

            if count % 100 == 0:
                print "Reading line %d" % count
newfile.close()
