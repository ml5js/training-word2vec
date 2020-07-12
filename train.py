# Code to train a word2vec model with gensim
# For use with ml5.js word2vec examples

import re
import json
import sys
import argparse
import glob
import os
from collections import Counter

from gensim.models import Word2Vec
import nltk

#Parsing for the user arguments
parser = argparse.ArgumentParser(description="Text File to Word2Vec Vectors")

#Required input file
parser.add_argument("input", help="Path to the input text file")

#Optional arguments (room for further extending the script's capabilities)
parser.add_argument("-o", "--output", default="vector.json", help="Path to the output text file (default: vector.json)")

# Let the user select a tokenizer
TOKENIZER_CHOICE_NLTK = 'nltk'
TOKENIZER_CHOICE_SIMPLE = 'simple'
TOKENIZER_CHOICES = [
    TOKENIZER_CHOICE_NLTK,
    TOKENIZER_CHOICE_SIMPLE
]
parser.add_argument("-t", "--tokenizer", default=TOKENIZER_CHOICE_SIMPLE, choices=TOKENIZER_CHOICES, help="Which tokenizer should be used.")
parser.add_argument("--remove-stop-words", action='store_true', help="Remove stopwords from the corpus.")
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

args = parser.parse_args()

#Using the arguments from the arg dictionary
output_text_file = args.output

listOfFiles = []
if os.path.isdir(args.input):
    # Make a list with all txt in the folder
    listOfFiles = glob.glob(args.input + '/*.txt')
else:
    # use a single file
    listOfFiles.append(args.input)

# If we're removing stop words then create a dictionary for faster lookup.
if args.remove_stop_words:
    # This is a bit of a messy setup but -- if we fail to have the stopwords
    # data then we download it and try again.
    try:
        stop_words = nltk.corpus.stopwords.words('english')
        stopwords_dict = Counter(stop_words)
    except LookupError:
        nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    stopwords_dict = Counter(stop_words)

# We also check that we have the correct tokenizer and download required
# supporting data if it not available
if args.tokenizer == TOKENIZER_CHOICE_NLTK:
    try:
        sentences = nltk.tokenize.sent_tokenize('Useless. Do we have punkt?')
    except LookupError:
        nltk.download('punkt')

# Generate a list of all of the sentences in files.
# Each sentence is an array of words.
final_sentences = []
for file in listOfFiles:
    text = open(file).read().lower().replace("\n", " ") # Remove lineabreaks

    # Remove all the stop words before running the actual tokenization.
    # I think it's a little bit cleaner to do it here and may perform because
    # there are fewer nested loops.
    if args.remove_stop_words:
        text = ' '.join([word for word in text.split() if word not in stopwords_dict])

    # Use NLTK's tokenizer
    if args.tokenizer == TOKENIZER_CHOICE_NLTK:
        sentences = nltk.tokenize.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.tokenize.word_tokenize(sentence)
            final_sentences.append(words)

    # Use a simple tokenizer
    else:
        sentences = re.split("[.?!]", text)
        # Split each sentence into words! (this could also be improved!)
        for sentence in sentences:
            words = re.split(r'\W+', sentence)
            final_sentences.append(words)


# Create the Word2Vec model
model = Word2Vec(final_sentences, size=100, window=5, min_count=5, workers=4)
# Save the vectors to a text file
model.wv.save_word2vec_format(output_text_file, binary=False)

# Open up that text file and convert to JSON
f = open(output_text_file)
v = {"vectors": {}}
for line in f:
    w, n = line.split(" ", 1)
    v["vectors"][w] = list(map(float, n.split()))

# Save to a JSON file
# Could make this an optional argument to specify output file
with open(output_text_file[:-4] + "json", "w") as out:
    json.dump(v, out)

