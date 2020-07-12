# Training

## Python Environment

## Requirements

- Set up a python environment with [gensim](https://radimrehurek.com/gensim/) installed. [More detailed instructions here](https://ml5js.org/docs/training-setup.html). You can also follow this [video tutorial about Python virtualenv](https://youtu.be/nnhjvHYRsmM).

```
pip install -r requirements.txt
```

## Train the model

1. Clone this repository or [download this python script](https://github.com/ml5js/training-word2vec/blob/master/train.py)

```
git clone https://github.com/ml5js/training-word2vec/
```

2. The script supports training from a single text file or directory of files. Create a text file or folder of multiple files. Now run `train.py` with the name of the file or folder.

Example:

```
python train.py file.xt
python train.py files/
```


3. The script will output a `vectors.txt` and `vectors.json` file, however, if you would like to specify an output file name you can use the additional argument `-o` for that.

```
python train.py data.txt -o output.json
```

4. The output JSON file can be used now with the [ml5.js word2vec examples](https://github.com/ml5js/ml5-examples/tree/master/p5js/Word2Vec).

## Advanced tokenization

The default tokenizer is very basic. You can ask the script to use NLTK's
tokenizer with the `--tokenizer` argument.

Additionally, the script can remove stop words.

```
python train.py files/ -t nltk --remove-stop-words
```

