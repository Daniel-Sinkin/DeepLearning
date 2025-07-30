# DeepLearning
Implementation of different deep learning specific things

# Implementations
* Encoder-Decoder Transformer in PyTorch, build to be aligned with Vasvani et al. 2017
* UNet based diffusion Model
* LeNet

# Visualisations
## Rejection Sampling
Suppose we want to sample a complex distribution $p(z)$ what is easy to evaluate if we ignore
normalisation, i.e., such that
$$
\tilde{p}(z) = C p(z)
$$
C > 0 is easy to evaluate. Take some simple to evaluate distribution $q(z)$ (for example normal)
and find $k > 0$ such that
$$
kq(z) > \tilde{p}(z)
$$
for all $z$. Then do uniform sampling on $\xi \sim U(0, kq(z_0))$ and accept if $\xi \leq \tilde{p}(z)$,
reject otherwise. The acceptance rate is proportional to $\frac{1}{k}$ so we want to have k
as small as possible.
![Rejection Sampling](visualisations/plots/rejection_sampling.png)
## Ridge Regression
Here I implemented the discussion in this stackexchange answer https://stats.stackexchange.com/a/151351
which shows the "Ridge" that we get with overfitting explicitly
![Ridge Regression Plot](visualisations/plots/ridge_regression.png)
## Curve Fitting
Assuming we are fitting a curve with Gaussian Noise, then for any fixed x value the distribution
of the target value is given as follows (based on Figure 1.28 in Bishop, *Pattern Recognition and Machine Learning*, 2006)
![Curve Fitting Gaussian](visualisations/plots/prob_regression_for_curve_fitting_plot.png)
## PCA
We take our samples $X = \{x_1, \dots, x_N\} \subseteq \mathbb{R}^D$ and construct the covariance matrix $\Sigma_X = \sum_{i = 1}^N x_i^\top x_i$.
This is a symmetric positive definite matrix (assuming all samples are linearly independent) as such
it has $N$ eigenvectors $v_1, \dots, v_N$ and $n$ positive (!) eigenvalues $\lambda_1, \dots, \lambda_N$,
without loss of generality we can assume that $\lambda_1 \geq \lambda_2 \geq \dots \lambda_N$.

The first $k$ eigenvectors then give us the most important  to extract the most important components
![PCA](visualisations/plots/pca_visualisation.png)
## Effect of regularisation
Suppose we have a covariance matrix $\Sigma \in \mathbb{R}^{N \times N}$, we can regularise by
making the diagonal more dominant, applying so-called Tychonoff (or Tikhonov) Regularisation: $\Sigma_\alpha = \alpha I_{N \times N} + \Sigma$
The effect of this is it makes the underlying equicontours more spherical, in the sense that
the eigenvalues are closer together ($\lambda_i / \lambda_{i + 1}$ goes closer to 1)
![Regularisation](visualisations/plots/covariance_regularisation.png)
## Box Muller
A somewhat efficient algorithm for sample standard normal random variable using a transform on
uniforms distributed variables. Suppose we want to sample $2N$ standard normal variables.

First start with $X^{(0)} := \{(x_i, y_i) : 1 \leq i \leq N\}$ random variables such that $x_i, y_i \sim U(0, 1)$
and reject all samples where $r_i^2 := x_i^2 + y_i^2 > 1.0,$
giving us $X^{(1)} = \{(x_i, y_i) \in X : r_i^2 \leq 1.0\}.$
![BM Square Disk](visualisations/plots/box_mueller/box_mueller_square_disk.png)
We then apply the transform $s_i^{(1)} = x_i \sqrt{\frac{-2\log(r_i^2)}{r_i^2}}$ and $s_i^{(2)} = y_i \sqrt{\frac{-2\log(r_i^2)}{r_i^2}}$ it can be shown that $s_i^{(1)}$ and $s_i^{(1)}$ are iid with $s_i^{(j)} \sim \mathcal{N}(0, 1)$.
![BM Square Disk](visualisations/plots/box_mueller/box_mueller_normals.png)
This uses the rotational symmetry of the 2 dimensional normal distribution. We can see more accurately
what the transformation does by highlighting a particular slice
![BM Slice](visualisations/plots/box_mueller/box_mueller_ring_highlight.png)
![BM Slice](visualisations/plots/box_mueller/box_mueller_ring_mapped.png)
# Integral Density
When doing Bayesian inference, if $p(\hat{\theta}|x)$ is very close to 1 for a particular $\hat{\theta}$
and close to $0$ for all others then $p(x|X) \approx p(X|\hat{\theta})$
This visualises this phenomonon
![Integral Density 0020 2600](visualisations/plots/integral_density/integral_density_var0020_theta2600.png)
![Integral Density 0330 2000](visualisations/plots/integral_density/integral_density_var0330_theta2000.png)
![Integral Density 1000 2650](visualisations/plots/integral_density/integral_density_var1000_theta2650.png)

## Gaussian Process
Normal
![Gaussian Process](visualisations/plots/gaussian_process/gaussian_process.png)
With added noise
![Gaussian Process Noise](visualisations/plots/gaussian_process/gaussian_process_noisy.png)

### Contribution Histograms
![Gaussian Process Hist 50](visualisations/plots/gaussian_process/gaussian_process_interactive_hist_50.png)
![Gaussian Process Hist 65](visualisations/plots/gaussian_process/gaussian_process_interactive_hist_65.png)
![Gaussian Process Hist 105](visualisations/plots/gaussian_process/gaussian_process_interactive_hist_105.png)

### Heatmap of different Kernels
![Gaussian Process Kernel Heatmap](visualisations/plots/gaussian_process/gaussian_process_kernel_heatmaps.png)

# References
* Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.