
import argparse
import json
import string
import sys

import nltk
import gensim.downloader

if __name__ == '__main__':

    # Parsing for the user arguments
    parser = argparse.ArgumentParser( description="Gensim Model to ML5JS word embedding json file. It will load a specified model and then (optionally) filter the vocabulary and generate a json file for use with ML5JS." )
    parser.add_argument("-o", "--output", default="out.json", help="Name of the json file to write the results to.")
    parser.add_argument("--pretty", action="store_true", help="If the json out should be prettified.")
    parser.add_argument("-b", "--base-model", default="glove-wiki-gigaword-50", help="Name of the model to take the word vectors from. See https://github.com/RaRe-Technologies/gensim-data for valid model names. (default='glove-wiki-gigaword-50')")
    parser.add_argument("-w", "--word-list-file", default="", help="Name of the file to take the desired words from. If no file is provided, it will use a fixed number of words. The file should have each word on its own line. (default='')")
    parser.add_argument("-s", "--remove-stop-words", action="store_true", help="If stop words (extremely common words that dont add much information) should be removed from the model. ")
    parser.add_argument("-p", "--remove-punctuation", action="store_true", help="If punctuation marks should be removed from the model.")
    parser.add_argument("-c", "--word-count", type=int, default="0", help="The maximum number of words to include in the resulting model. (default=1000)")
    parser.add_argument("--top-level-key", default="vectors", help="The top most key in the output json file. For ML5JS this should be 'vectors' (default). You should not need to change this.")

    args = parser.parse_args()

    # Use aliases for the configuration parameters so they are easier to read
    # in the code (and I dont have to put args.XXX everywhere).
    CONFIG_GENSIM_BASE_MODEL = args.base_model
    CONFIG_WORD_LIST_FILE = args.word_list_file
    CONFIG_REMOVE_STOP_WORDS = args.remove_stop_words
    CONFIG_REMOVE_PUNCTUATION = args.remove_punctuation
    CONFIG_WORD_COUNT = args.word_count
    CONFIG_OUTPUT_FILENAME = args.output
    CONFIG_OUTPUT_PRETTY = args.pretty
    CONFIG_TOP_LEVEL_KEY = args.top_level_key

    # If the user has supplied a file to be used a word list - load it.
    wordList = []
    if CONFIG_WORD_LIST_FILE:
        print( 'Loading word list from "%s" ...' % CONFIG_WORD_LIST_FILE )
        with open( CONFIG_WORD_LIST_FILE ) as f:
            wordList = f.read().splitlines()
        print( 'Loaded %d words from word list file' % len(wordList) )

    # Load the gensim model that we will use to get the word vectors from.
    # If there is error loading the model: exit and tell the user.
    if CONFIG_GENSIM_BASE_MODEL:
        print( 'Using base model: %s' % CONFIG_GENSIM_BASE_MODEL )
        try:
            model = gensim.downloader.load( CONFIG_GENSIM_BASE_MODEL )
        except Exception:
            print( 'Failed to download model. Is your model name valid? See https://github.com/RaRe-Technologies/gensim-data for valid model names' )
            print( 'See the exception/stack trace printed below for more info.' )
            raise

    # If there is no model: exit and tell the user we need a model.
    else:
        print( 'Please specify a base model name, for more info see https://github.com/RaRe-Technologies/gensim-data' )
        sys.exit(1)


    # When the user doesnt provide a word list we will use ALL or some of the
    # of words in the vocabulary of the model, based on the other parameters.
    if not wordList:
        print( 'No word list provided, using first %d words from the base model' % CONFIG_WORD_COUNT )

        # Load the stop words if asked, and tell the user
        if CONFIG_REMOVE_STOP_WORDS:
            print( 'Removing stopwords (as defined by nltk)' )
            sw = nltk.corpus.stopwords.words('english')

        # Tell the user if we are removing punctuation
        if CONFIG_REMOVE_PUNCTUATION:
            print( 'Removing punctuation' )

        # If the word count is 0 -- use all the words in the model's vocabulary
        if CONFIG_WORD_COUNT == 0:
            print( 'Attempting to use all %d words in the vocabulary' % len(model.vocab) )
            CONFIG_WORD_COUNT = len(model.vocab)

        # Go through as many words as needed in the vocabulary to generate a
        # list of words we want
        wordIndex = 0
        for word in model.vocab:

            # Skip stop words
            if CONFIG_REMOVE_STOP_WORDS and word in sw:
                continue

            # Skip punctuation
            if CONFIG_REMOVE_PUNCTUATION:
                if word in string.punctuation:
                    continue
                if word in [ "''", '``', '""' ]:
                    continue

            # Add the word to the list
            wordList.append( word )
            wordIndex += 1

            # Stop once we have enough words
            if wordIndex >= CONFIG_WORD_COUNT:
                break

    else:
        print( 'Using the supplied word list to filter words' )

    print( 'Word list prepared, building json data to export...' )


    # Build a json object that matches the structure used by ML5JS.
    # Alert the user to any words that fail to be added.
    jsonExport = {}
    jsonExport[CONFIG_TOP_LEVEL_KEY] = {}
    for word in wordList:
        try:
        	jsonExport[CONFIG_TOP_LEVEL_KEY][word] = model[word].tolist()
        except Exception:
            print( 'Failed to use word "%s"' % word )

    # Write the actual output file.
    with open( CONFIG_OUTPUT_FILENAME, 'w' ) as f:
        if CONFIG_OUTPUT_PRETTY:
    	    f.write( json.dumps(jsonExport,indent=2) )
        else:
    	    f.write( json.dumps(jsonExport) )

    # Done!
    print( 'Done. See %s for results.' % CONFIG_OUTPUT_FILENAME )


