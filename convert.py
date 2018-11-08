# Code to convert an existing word2vec model in txt format
# to json format for use with ml5

import re
import json
import sys

path = sys.argv[1]

# Open up that text file and convert to JSON
f = open(path)
v = {"vectors": {}}
for line in f:
    w, n = line.split(" ", 1)
    v["vectors"][w] = list(map(float, n.split()))

# Save to a JSON file
# Could make this an optional argument to specify output file
with open("vectors.json", "w") as out:
    json.dump(v, out)
