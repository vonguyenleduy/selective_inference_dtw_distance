# Statistical Inference for the Dynamic Time Warping Distance, with Application to Abnormal Time-Series Detection

We study statistical inference on the similarity/distance between two time-series under uncertain environment by considering a statistical hypothesis test on the distance obtained from Dynamic Time Warping (DTW) algorithm. The sampling distribution of the DTW distance is too complicated to derive because it is obtained based on the solution of a complicated algorithm. To circumvent this difficulty, we propose to employ a conditional sampling distribution for the inference, which enables us to derive an exact (non-asymptotic) inference method on the DTW distance. Besides, we also develop a novel computational method to compute the conditional sampling distribution. To our knowledge, this is the first method that can provide valid $p$-value to quantify the statistical significance of the DTW distance, which is helpful for high-stake decision making.



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


