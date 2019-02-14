# Multilingual Neural Machine Translation with Soft Decoupled Encoding
This is the code we used in our paper
>[Multilingual Neural Machine Translation with Soft Decoupled Encoding](https://arxiv.org/pdf/1902.03499.pdf)

>Xinyi Wang, Hieu Pham, Philip Arthur, Graham Neubig


## Requirements

Python 3.6, PyTorch 0.4.1


All the scripts for experiments in the paper can be created from the templates under scripts/template/

## Data Processing

The data we use is [multilingual TED corpus](https://github.com/neulab/word-embeddings-for-nmt) by Qi et al.

We provide preprocessed version of the data, which you can get from:
If you are interested int the details of data processing, you can take a look at the script ``make-eng.sh`` and  ``make-data.sh``.

## Training:
The template name for the following methods are:
  1. SDE: bi-semb-bq-o32000
  2. subword: bi-sw-32000
  2. subword-joint: bi-sw-joint-32000
  3. word: bi-w-64000

To make the main experiment scripts for alll 4 languages tested in the paper, simply call
``bash make-cfg.sh``

## Decoding:
To make decode scripts, simply use the file make-trans.py. Change the name of the directory where the experiment outputs are stored if you modify the template scripts during training. Otherwise it should just work by calling:
``python make-trans.py``

