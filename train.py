# Code to train a word2vec model with gensim
# For use with ml5.js word2vec examples

from gensim.models import Word2Vec
import re
import json
import sys

# Path to the text file is the second argument
path = sys.argv[1]
text = open(path).read().lower().replace("\n", " ")

# Split into sentences (this could be improved! Using nltk?)
sentences = re.split("[.?!]", text)

# Split each sentence into words! (this could also be improved!)
final_sentences = []
for sentence in sentences:
    words = re.split("\W+", sentence)
    final_sentences.append(words)

# Create the Word2Vec model
model = Word2Vec(final_sentences, size=100, window=5, min_count=5, workers=4)
# Save the vectors to a text file 
model.wv.save_word2vec_format('vectors.txt', binary=False)

# Open up that text file and convert to JSON
f = open("vectors.txt")
v = {"vectors": {}}
for line in f:
    w, n = line.split(" ", 1)
    v["vectors"][w] = list(map(float, n.split()))

# Save to a JSON file
# Could make this an optional argument to specify output file
with open("vectors.json", "w") as out:
    json.dump(v, out)
