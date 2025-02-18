# Methodology

## Nadaraya-Watson Estimator
Suppose we have data $(x_i,y_i)_{i=1}^N$ and a kernel function $\mathbf{k}$, the Nadaraya-Watson estimate at $x$ is defined as
$$
\widehat{f}(x) := \frac{\sum_{i=1}^N \mathbf{k}(x,x_i) y_i}{\sum_{i=1}^N \mathbf{k}(x,x_i)}.
$$