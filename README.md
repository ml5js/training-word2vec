# Training

## Python Environment

## Requirements

- Set up a python environment with [gensim](https://radimrehurek.com/gensim/) installed. [More detailed instructions here](https://ml5js.org/docs/training-setup.html). You can also follow this [video tutorial about Python virtualenv](https://youtu.be/nnhjvHYRsmM).

```
pip install gensim
```

## Train the model

1. Clone this repository or [download this python script](https://github.com/ml5js/training-word2vec/blob/master/train.py)

```
git clone https://github.com/ml5js/training-word2vec/
```

2. The script supports training from many files. 
Create a new folder and copy your text file into this directory. Now run `train.py` with the name of the folder.

Example:

```
python train.py inputs/
```
3. If you would like to add an output file path you can use the additional argument `-o` for that.

```
python train.py data.txt -o output.json
```

4. The script will save a file called `vectors.json`, or whatever you passed as the output argument. You can then use this file the [ml5.js word2vec examples](https://github.com/ml5js/ml5-examples/tree/master/p5js/Word2Vec).
