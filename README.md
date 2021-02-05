# Group Testing with Features

Repository for my project for the class Mathematics for Datascience (ENS Ulm, Master MVA, Gabriel Peyré). 

## Short presentation of the project

My project aims at introducing prior knowledge about tested individuals, in the form of features, in the traditionnal setting of group testing. It fits a narrative in which a person arriving in a testing facility is first asked some questions about her behaviour, symptoms, etc. and may then be group tested if the tester is unsure of her status. I test three different ways to include prior knowledge: 

1. In hard thresholding, I train a basic classifier (a logistic regression) on a training set, run it on the test set and eliminate from the group testing procedure all individuals whose status is predicted with confidence superior to a certain threshold.
2. In soft thresholding, the testing matrix is constructed as a function of the probabilities of being positive for each individual.
3. In soft hard thresholding, some individuals are randomly excluded from the testing procedure based on their probability of being positive.

See the companion pdf file for details and simulations. Simulations are only made on synthetic datasets. 

## Decoding algorithms 

This projection containts an implementation of some decoding algorithms used in group testing (see [this article](https://ieeexplore.ieee.org/abstract/document/6781038)), contained in the file `reconstruction_alg.py`. The file `utils.py` contains some useful utility functions for creating CTPI and Bernoulli tests. Feel free to use the decoding algorithms (but beware, the code is really slow).
