import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils import shuffle


color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)


    plt.title(title)
    plt.show()


x_mu1 = [-2.0, -2.0]
x_sigma1 = [[0.2, 0.3], [0.3, 0.5]]

x_mu2 = [1.0, 1.0]
x_sigma2 = [[0.4, 0.2], [0.2, 0.2]]

x_data1 = np.random.multivariate_normal(x_mu1, x_sigma1, 1000).T
x_data2 = np.random.multivariate_normal(x_mu2, x_sigma2, 1000).T
x_data = np.hstack((x_data1, x_data2))
x1, x2 = x_data

###################
# Initial parameters
alpha_0 = 0
w0 = 0
v0 = 0
mu0 = 0
beta0 = 0

#########################

x1, x2 = shuffle(x1, x2)


model = BayesianGaussianMixture(n_components=2,
                                covariance_type="full",
                                max_iter=100,
                                init_params="random",
                                weight_concentration_prior_type="dirichlet_process",
                                verbose=1, verbose_interval=10)
data = np.asarray([x1, x2]).T


model.fit(data)
print(model.get_params())

plot_results(data, model.predict(data), model.means_, model.covariances_, 0, 'GMM')

"""
plt.figure()
plt.plot(x1, x2, '.')
plt.show()
"""