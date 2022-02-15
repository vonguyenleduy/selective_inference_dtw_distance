# Exact Statistical Inference for Time Series Similarity using Dynamic Time Warping by Selective Inference

This package implements an exact statistical inference for the DTW Distance by Selective Inference.

## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

We recommend to install or update anaconda to the latest version and use Python 3
(We used Python 3.8.3).

## Examples

#### (1) Example of computing selective p-value
```
>> python ex1_compute_p_value.py
```

#### (2) Checking the uniformity of the selective p-value

To check the uniformity of the p-value, please run  
```
>> python ex2_uniform_p_value.py
```
This is an option to check the correctness of the proposed method. The proposed selective-p value must follow uniform distribution. 


