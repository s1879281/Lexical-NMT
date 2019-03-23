
# Lexical-Neural-Machine-Translation

This is a Pytorch implementation of [Improving Lexical Choice in Neural Machine Translation](https://arxiv.org/pdf/1710.01329.pdf).

![visualization](https://github.com/s1879281/Lexical-NMT/blob/master/images/vis.jpg)

## Installing packages
To run the code, you should have all the required packages installed.

* Install pyTorch 0.4.1
```
$> pip install -U pip
$> pip install https://bit.ly/2tkNNuO
```
* Install additional packages required by the code
```
$> pip install tqdm seaborn pandas matplotlib numpy
```

## Baseline NMT model
You’ll find several directories,including raw data containing raw English and Japanese parallel data (from this helpful [tutorial](https://github.com/neubig/nmt-tips),
prepared data containing the pre-processed data your models will be trained on, and seq2seq containing the encoder-decoder model.
* *train.py* is used to train the translation models.
* *translate.py* translates the test-set greedily using model parameters
 restored from the best checkpoint file and saves the output to *model_translations.txt*.
* *visualize.py* generates heat-maps from the decoder-to-encoder attention weights for the first 10 sentence pairs in the test-set.

## Lexical Model
Train the translation model after augmenting it with the lexical model by running the following command:
```
python train.py --decoder-use-lexical-model True
```

## Evaluation
Use the following command to calculate the test-BLEU score:
```
python translate.py
perl multi-bleu.perl -lc raw_data/test.en < model_translations.txt
```

## Reference

If you use this code as part of any published research, please acknowledge the following paper:

```
@article{nguyen2017improving,
  title={Improving lexical choice in neural machine translation},
  author={Nguyen, Toan Q and Chiang, David},
  journal={arXiv preprint arXiv:1710.01329},
  year={2017}
}
```