# Training

## Python Environment

## Requirements

- Set up a python environment with [gensim](https://radimrehurek.com/gensim/) installed. [More detailed instructions here](https://ml5js.org/docs/training-setup.html). You can also follow this [video tutorial about Python virtualenv](https://youtu.be/nnhjvHYRsmM).

```
pip install gensim
```

- If you are familiar with Docker, you can also use this  ~~[container]()~~ (soon!)


## Train the model

1. Clone this repository or [download this python script](https://github.com/ml5js/training-word2vec/blob/master/train.py)

```
git clone https://github.com/ml5js/training-word2vec/
```

2. The script in its current form only supports training from a single text file. Copy your text file into this directory and run `train.py` with the name of the file.

```
python train.py data.txt
```

3. The script will save a file called `vectors.json`. You can then use this file the [ml5.js word2vec examples](https://github.com/ml5js/ml5-examples/tree/master/p5js/Word2Vec).
