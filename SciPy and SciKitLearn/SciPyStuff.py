import numpy as np
import pylab as pl
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, svm, neighbors


class SciPyStuff:
    def __init__(self):
        np.set_printoptions(suppress=True)   # Stops scientific notation

        #self.norm()
        #self.gamma()
        #self.exponential()
        #self.integration()
        self.learning()

    def norm(self):
        a = np.random.normal(size=1000)
        bins = np.arange(-4, 5)
        histogram = np.histogram(a, bins=bins, normed=True)[0]
        bins = 0.5 * (bins[1:] + bins[:-1])
        #print(bins)

        b = stats.norm.pdf(bins)

        plt.plot(bins, histogram)
        plt.plot(bins, b)
        sns.plt.show()

    def gamma(self):
        a = np.random.gamma(size=1000, shape=1)
        bins = np.arange(-4,5)
        print('bins: ', bins)
        histogram = np.histogram(a, bins=bins, normed=True)[0]
        print('histogram: ', histogram)
        bins = 0.5 * (bins[1:] + bins[:-1])
        print('bins: ', bins)

        b = stats.gamma.pdf(bins, 1)

        plt.plot(bins, histogram)
        plt.plot(bins, b)
        sns.plt.show()

    def exponential(self):
        a = np.random.exponential(size=1000)
        b = stats.expon.pdf(a)
        plt.plot(a, b)

        sns.plt.show()

    def integration(self):
        res, err = integrate.quad(np.sin, 0, np.pi/2)   # quad() is most common
        print(res, err)

        # # Solve the ODE dy/dt = -2y between t= 0, ..., 4, with initial condition y(t=0) = 1.
        # counter = np.zeros((1,), dtype = np.uint16)
        #
        # # Trajectory will be computed:
        # time_vec = np.linspace(0, 4, 40)
        # yvec, info = integrate.odeint(self.calc_derivative(1, time_vec, args = (counter,)), full_output=True)

    def calc_derivative(self, ypos, time, counter_arr):
        counter_arr += 1
        return -2 * ypos

    def learning(self):
        iris = datasets.load_iris()
        digits = datasets.load_digits()
        print(digits.images.shape)

        pl.imshow(digits.images[1], cmap=pl.cm.gray_r)
        #pl.show()

        # Translate each 8x8 image into a vector of length 64 to work with the data:
        data = digits.images.reshape((digits.images.shape[0], -1))

        # In Scikit, we learn from data by creating an estimator and calling its fit(X,Y) method.
        clf = svm.LinearSVC()
        print(clf.fit(iris.data, iris.target))

        # Once we have learned from our data, we can use our model to predict the most likely outcome.
        print(clf.predict([[5.0, 3.6, 1.3, 0.25]]))
        print(clf.coef_)    # Parameters of the model
        print()

        # Classification
        # k-Nearest Neighbors Classifier
        knn = neighbors.KNeighborsClassifier()
        knn.fit(iris.data, iris.target)
        print(knn.predict([[0.1, 0.2, 0.3, 0.4]]))

        # Training set and testing set
        # With knn estimator, we get perfect prediction on the training set.
        perm = np.random.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]
        print(knn.fit(iris.data[:100], iris.target[:100]))
        print(knn.score(iris.data[:100], iris.target[:100])) # Probability? Idk.
        print()

        # Support Vector Machines(SVMs) for classification
        # Linear SVMs
        svc = svm.SVC(kernel='linear')
        svc.fit(iris.data, iris.target)
        print(svc.predict([[0.1, 0.2, 0.3, 0.4]]))
        print(svc.score(iris.data[:100], iris.target[:100]))

SciPyStuff()