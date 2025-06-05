---
title: 'N$^2$: A unified Python package and test bench for nearest neighbor-based matrix completion'
tags:
  - Python
  - matrix completion
  - nearest neighbors
  - causal inference
authors:
  - name: Caleb Chin
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: Aashish Khubchandani
    orcid: 0009-0003-2638-9472
    corresponding: true
    affiliation: 1
  - name: Harshvardhan Maskara
    affiliation: 1
  - name: Kyuseong Choi
    orcid: 0000-0002-3380-2849
    corresponding: true
    affiliation: 1
  - name: Jacob Feitelberg
    orcid: 0000-0002-4551-0245
    corresponding: true
    affiliation: 2
  - name: Albert Gong
    orcid: 0009-0005-1687-0240
    corresponding: true
    affiliation: 1
  - name: Manit Paul
    orcid: 0000-0002-1735-3289
    affiliation: 3
  - name: Tathagata Sadhukhan
    orcid: 0009-0001-1549-1469
    affiliation: 1
  - name: Anish Agarwal
    affiliation: 2
  - name: Raaz Dwivedi
    orcid: 0000-0002-9993-8554
    affiliation: 1
affiliations:
 - name: Cornell University, USA
   index: 1
 - name: Columbia University, USA
   index: 2
 - name: University of Pennsylvania, USA
   index: 3
date: 28 May 2025
bibliography: ref.bib

---

# Summary

N$^2$ is a Python package that provides a unified framework for nearest neighbor-based matrix completion methods as well as a test bench for evaluating the performance of these methods. It includes implementation of several nearest neighbor (NN) methods including row-wise NN, column-wise NN, two-sided NN, and doubly-robust NN. Additionally, N$^2$ includes implementations for new distributional nearest neighbor algorithms which impute distributions instead of just scalars. The test bench included with the package allows researchers and data scientists to easily evaluate the performance of these methods and future variants on heterogeneous real-world datasets. The package is designed to be simple to use and extendible, allowing users to easily customize the algorithms and test different methods on their own datasets.

# Statement of need

Nearest neighbor methods are powerful and simple non-parametric algorithms used in a variety of machine learning and pattern recognition tasks. Their scalability and simplicity have also made them popular choices for matrix completion and causal inference tasks in panel-data settings. In particular, nearest neighbor methods have shown robustness to a variety complex of missingness patterns common in real-world datasets. This has led to a multitude of recent works extending the basic nearest neighbors algorithm beyond the standard row-wise nearest neighbor approach. However, there is a lack of a unified framework for implementing and evaluating these new methods. N$^2$ fills this gap by implementing a variety of nearest neighbor-based matrix completion methods and providing a test bench for evaluating their performance. This test bench also provides users a unified framework to test various matrix completion methods on their own datasets, something that was previously difficult to do because each method had its own specific implementation and interface.

There are other existing packages which implement nearest neighbor methods such as `scikit-learn` and `fancyimpute`, but these packages do not provide a unified framework for matrix completion methods. Additionally, `scikit-learn` and `fancyimpute`focuses only on row-wise nearest neighbor for supervised learning tasks. Neither of these packages provide a test bench for evaluating the performance of new methods and do not include implementations for the new distributional nearest neighbor algorithms.

# Unified framework

# Test bench

# Acknowledgements

Jacob Feitelberg and Anish Agarwal were supported by the Columbia Center for AI and Responsible Financial Innovation in collaboration with Capital One. Albert Gong's work was partially supported by funding from NewYork-Presbyterian for the NYP-Cornell Cardiovascular AI Collaboration.

# References
