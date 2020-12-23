# mSIMPAD
Multiple length Successive sIMilar PAtterns Detector (mSIMPAD), is a method for detecting Successive Similar Pattern (SSP) - a series of similar sequences that occur consecutively at non-regular intervals in time series. The method is based on the Matrix Profile, a powerful tool for time series mining (we encourge interested individuals to learn more at <a href="https://www.cs.ucr.edu/~eamonn/MatrixProfile.html">the UCR Matrix Profile Page</a>).

If you find this code useful in your research, please consider citing:
```
@article{10.1145/3396250, author = {Li, Chun-Tung and Cao, Jiannong and Liu, Xue and Stojmenovic, Milos}, title = {MSIMPAD: Efficient and Robust Mining of Successive Similar Patterns of Multiple Lengths in Time Series}, year = {2020}, issue_date = {November 2020}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, volume = {1}, number = {4}, issn = {2691-1957}, url = {https://doi.org/10.1145/3396250}, doi = {10.1145/3396250}, abstract = {A successive similar pattern (SSP) is a series of similar sequences that occur consecutively at non-regular intervals in time series. Mining SSPs could provide valuable information without a priori knowledge, which is crucial in many applications ranging from health monitoring to activity recognition. However, most existing work is computationally expensive, focuses only on periodic patterns occurring in regular time intervals, and is unable to recognize patterns containing multiple periods. Here we investigate a more general problem of finding similar patterns occurring successively, in which the similarity between patterns is measured by the z-normalized Euclidean distance. We propose a linear time, robust method, called Multiple-length Successive sIMilar PAtterns Detector (mSIMPAD), that mines SSPs of multiple lengths, making no assumptions regarding periodicity. We apply our method on the detection of repetitive movement using a wearable inertial measurement unit. The experiments were conducted on three public datasets, two of which contain simple walking and idle data, whereas the third is more complex and contains multiple activities. mSIMPAD achieved F-score improvements of 3.2% and 6.5%, respectively, over the simple and complex datasets compared to the state-of-the-art walking detector. In addition, mSIMPAD is scalable and applicable to real-time applications since it operates in linear time complexity.}, journal = {ACM Trans. Comput. Healthcare}, month = sep, articleno = {23}, numpages = {19}, keywords = {matrix profile, repetitive movement detection, periodicity detection, Successive similar pattern} }
```

# To run the code

The code is implemented in Python and tested on Python 3.7. The `SSPDetector` has two methods: `SIMPAD(seq, l, m)` where seq is a [d x n] real value array, l is the target pattern length and m is a user-defined searching range; and `mSIMPAD(seq, L, m_factor)` where L is a list of target pattern lengths and m_factor is the multiplying factor of searching range as l * m_factor. We included all the implemented code and code that are publicly avaliable as well as the data. Please include the requested citation of the original authors if you are using their code and data.

Perform SSP detection as simplay:

```
import SSPDetector
detection = SSPDetector.SIMPAD(seq, l, m)
```

You may follow the steps below to execute the experiments:

1. Download the cached datasets <a href="https://connectpolyu-my.sharepoint.com/:u:/g/personal/14902288r_connect_polyu_hk/ETvjTjzj98FGv1N2HYZg9KoBgq5jpmz7DFvHRRQswe1T4Q?e=MzXtLl">here</a>.
2. Unzip the `data.zip` to the parent directory of this project.
3. Execute seperate experiments `experiment_ubicomp13.py`, `experiment_HAPT.py`, `experiment_PAMAP2.py`, and `experiment_speed.py`.

# Data

3 Datasets are included:

<b><a href="http://www.cl.cam.ac.uk/~ab818/ubicomp2013.html">UbiComp'13 Walking</a></b>
</br>
<cite>Harle, R., & Brajdic, A. (2017). Research data supporting "Walk detection and step counting on unconstrained smartphones" [Dataset]. https://doi.org/10.17863/CAM.12982</cite>
<br>

<b><a href="https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones">HAPT</a></b>
</br>
<cite>Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.</cite>
<br>

<b><a href="https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring">PAMAP2</a></b>
</br>
<cite>A. Reiss and D. Stricker. Introducing a New Benchmarked Dataset for Activity Monitoring. The 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.</cite>
<br>
<cite>A. Reiss and D. Stricker. Creating and Benchmarking a New Dataset for Physical Activity Monitoring. The 5th Workshop on Affect and Behaviour Related Assistance (ABRA), 2012.</cite>
