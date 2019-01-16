# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.gca().set_aspect('equal', adjustable='box')

from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

n = 1000

x = 2*np.random.rand(n, 2)-1
a, b = 0.85, 0.15
def filt_func(row):
	x, y = row
	out = (x**2 / a**2) + (y**2 / b**2) ## Equation of an ellipsoid
	if out > 2:
		return False
	else:
		return True
X = np.array(list(filter(filt_func, x)))


x1 = np.random.rand(n)
x2 = x1**2 + 0.1*np.random.rand(n)
X = np.array([x1, x2]).T


# theta = np.pi * np.random.rand()/2
# M = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
# X = np.matmul(X, M)


def plot2d(X, title):
	'''3d plot first 3 components of a data frame'''
	fig = plt.figure(1, figsize=(4, 3))
	plt.gca().set_aspect('equal')
	plt.scatter(X[:, 0], X[:, 1], cmap=plt.nipy_spectral,
	           edgecolor='k')
	plt.title(title)
	plt.show()

plot2d(X, 'original data')

pca = decomposition.PCA()
pca.fit(X)
X = pca.transform(X)

plot2d(X, 'PCA TRANSFORMED DATA')

print('PCA Components (column vectors are the eigen vectors)\n{}'.format(pca.components_))
print('Explained Variance Ratio\n{}'.format(pca.explained_variance_ratio_))

X[:,1] = 0
plot2d(pca.inverse_transform(X), 'INVERSE_TRANSFORM OF PCA DATA WITH SECOND COMPONENT REMOVED')

