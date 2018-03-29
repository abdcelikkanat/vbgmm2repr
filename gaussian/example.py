import numpy as np
import matplotlib.pyplot as plt



def show(x):
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], '.')
    plt.show()


def generate(num_of_samples=1000, dim=2, num_of_classes=2):

    mean_1 = [-1.0, -1.0]
    cov_1 = [[2.0, 1.0], [1.0, 5.0]]

    mean_2 = [5.0, 5.0]
    cov_2 = [[5.0, -3.0], [-3.0, 5.0]]


    x1 = np.random.multivariate_normal(mean_1, cov_1, 100)
    x2 = np.random.multivariate_normal(mean_2, cov_2, 100)

    x = np.vstack((x1, x2))
    np.random.shuffle(x)
    return x

def variationalBayes(x):


def main():

    x = generate()
    show(x)
    print(x)

main()