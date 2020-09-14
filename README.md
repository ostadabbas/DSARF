# Deep Switching Auto-Regressive Factorization (DSARF)

This repository is the implementation of our paper in the following arXiv preprint:

@article{farnoosh2020,
  title={Deep Switching Auto-Regressive Factorization: Application to Time Series Forecasting},
  author={Farnoosh, Amirreza and Azari, Bahar and Ostadabbas, Sarah},
  journal={https://arxiv.org/pdf/2009.05135.pdf},
  year={2020}
}


### Dependencies: 
Numpy, Scipy, Pytorch, Tqdm, Matplotlib, Sklearn, Json, Pandas

**Run the following snippets to restore results (from checkpoints) for each dataset for short- and long-term predictions respectively. For short-term prediction,  remove `-restore` to train from scratch, then add `-predict` to perform short-term prediction on the test set. For long-term prediction, remove `-restore` and add `-long` to train from scratch and perform long-term prediction on the test set.**

## Birmingham
Short-term: 
`python dsarf.py -k 10 -file ./data/birmingham.mat -smod ./checkpoints/birmingham/short/ -dpath ./results_birmingham_short/ -ID birmingham -last 7 -lag 1 2 -epoch 500 -bs 30 -restore`

Long-term: 
`python dsarf.py -k 10 -file ./data/birmingham.mat -smod ./checkpoints/birmingham/long/ -dpath ./results_birmingham_long/ -ID birmingham -last 7 -lag 1 2 3 18 19 20 126 127 128 -epoch 500 -bs 30 -long -restore`

## Guangzhou
Short-term: 
`python dsarf.py -k 30 -file ./data/guangzhou.mat -smod ./checkpoints/guangzhou/short/ -dpath ./results_guangzhou_short/ -ID guangzhou -last 5 -lag 1 2 -epoch 500 -bs 100 -restore`

Long-term: `python dsarf.py -k 30 -file ./data/guangzhou.mat -smod ./checkpoints/guangzhou/long/ -dpath ./results_guangzhou_long/ -ID guangzhou -lag 1 2 3 144 145 146 1008 1009 1010 -epoch 500 -bs 30  -last 5 -long -restore`

## Hangzhou 
Short-term: 
`python dsarf.py -k 10 -file ./data/hangzhou.mat -smod ./checkpoints/hangzhou/short/ -dpath ./results_hangzhou_short/ -ID hangzhou -last 5 -lag 1 2 -epoch 1000 -bs 25 -restore`

Long-term: 
`python dsarf.py -k 10 -file ./data/hangzhou.mat -smod ./checkpoints/hangzhou/long/ -dpath ./results_hangzhou_long/ -ID hangzhou -last 5 -lag 1 2 3 108 109 110 756 757 758 -epoch 1000 -bs 25 -long -restore`

## Seattle
Short-term: `
python dsarf.py -k 30 -file ./data/seattle.npz -smod ./checkpoints/seattle/short/ -dpath ./results_seattle_short/ -ID seattle -last 5 -lag 1 2 -epoch 500 -bs 1000 -restore`

long-term: 
`python dsarf.py -k 30 -file ./data/seattle.npz -smod ./checkpoints/seattle/long/ -dpath ./results_seattle_long/ -ID seattle -last 5 -lag 1 2 3 288 289 290 2016 2017 2018 -epoch 500 -bs 1000 -long -restore`

## Pacific Surface Temprature (PST)
Short-term: 
`python dsarf.py -k 50 -file ./data/pacific.tsv -smod ./checkpoints/pacific/short/ -dpath ./results_pacific_short/ -ID pacific -lag 1 -epoch 500 -bs 100 -last 5 -s 2 -restore`

Long-term: 
`python dsarf.py -k 50 -file ./data/pacific.tsv -smod ./checkpoints/pacific/long/ -dpath ./results_pacific_long/ -ID pacific -lag 1 2 12 13 84 85 -epoch 500 -bs 100 -last 5 -long -restore`

## Google flu
Short-term: 
`python dsarf.py -k 10 -file ./data/google_flu.txt -smod ./checkpoints/flu/short/ -dpath ./results_flu_short/ -ID flu -lag 1 2 3 4 -epoch 1000 -bs 1 -last 2 -restore`

Long-term: `
python dsarf.py -k 10 -file ./data/google_flu.txt -smod ./checkpoints/flu/long/ -dpath ./results_flu_long/ -ID flu -lag 1 2 52 53 104 105 -epoch 1000 -bs 1 -last 2 -long -restore`

## Google dengue
Short-term: 
`python dsarf.py -k 5 -file ./data/google_dengue.txt -smod ./checkpoints/dengue/short/ -dpath ./results_dengue_short/ -ID dengue -lag 1 2 -epoch 500 -bs 1 -last 2 -restore`

Long-term: 
`python dsarf.py -k 5 -file ./data/google_dengue.txt -smod ./checkpoints/dengue/long/ -dpath ./results_dengue_long/ -ID dengue -lag 1 2 52 53 104 105 -epoch 500 -bs 1 -last 2 -long -restore`

## Colorado Precipitation
Short-term:
`python dsarf.py -k 20 -file ./data/precipitation.json -smod ./checkpoints/precipitation/short/ -dpath ./results_precipitation_short/ -ID prec -lag 1 2 -epoch 500 -bs 1 -last 1 -s 3 -restore`

## Bat flight
Short-term: 
`python dsarf.py -k 5 -s 2 -lag 1 2 -epoch 500 -bs 1 -last 2 -ID bat -file ./data/bat.json -smod ./checkpoints/bat/short/ -dpath ./results_bat_short/ -restore`

## Apnea
Short-term: 
`python dsarf.py -k 1 -file ./data/apnea.txt -smod ./checkpoints/apnea/short/ -dpath ./results_apnea_short/ -lag 1 -epoch 2000 -bs 1 -s 2 -last 1 -ID apnea -restore`

## Lorenz attractor
`python dsarf_lorenz.py -k 3 -s 2 -lag 1 -epoch 5000 -bs 100 -file ./data/lorenz.json -smod ./checkpoints/lorenz/ -dpath ./results_lorenz/ -restore`

## Double pendulum
`python dsarf_pendulum.py -k 4 -s 3 -lag 1 2 -epoch 500 -bs 100 -file ./data/pendulum.json -smod ./checkpoints/pendulum/ -dpath ./results_pendulum/ -last 2 -restore`

## Toy example
`python dsarf_toy.py -k 2 -s 2 -lag 1 2 3 -epoch 500 -bs 100 -file ./data/toy.json -smod ./checkpoints/toy/ -dpath ./results_toy/ -restore`
