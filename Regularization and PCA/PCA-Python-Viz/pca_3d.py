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

x = 2*np.random.rand(n, 3)-1
a, b, c = 0.85, 0.5, 0.25
def filt_func(row):
	x, y, z = row
	out = (x**2 / a**2) + (y**2 / b**2) + (z**2/c**2) ## Equation of an ellipsoid
	if out > 2:
		return False
	else:
		return True
X = np.array(list(filter(filt_func, x)))
M = np.random.rand(3, 3)
X = np.matmul(X, M)

x1 = np.random.rand(n)
x2 = 5*x1 + 3*np.random.rand(n)
x3 = 2*x1**2 + 3*np.random.rand(n)
X = np.array([x1, x2, x3]).T

def plot3d(X, title):
	'''3d plot first 3 components of a data frame'''
	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	plt.cla()

	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)

	plt.gca().set_aspect('equal', adjustable='box')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.nipy_spectral,
	           edgecolor='k')
	plt.title(title)
	plt.show()

plot3d(X, 'Raw Data')

pca = decomposition.PCA(3)
pca.fit(X)
print(pca)
X = pca.transform(X)
plot3d(X, 'PCA Transformed Data')
print('PCA Components (column vectors are the eigen vectors)\n{}'.format(pca.components_))
print('Explained Variance Ratio\n{}'.format(pca.explained_variance_ratio_))
print('Relative Variance Ratio\n{}'.format(np.cumsum(pca.explained_variance_ratio_)))

plot3d(pca.inverse_transform(X), 'PCA Inverse Transformed Data')

x = X.copy()
x[:,2] = 0

plot3d(pca.inverse_transform(x), 'PCA Inverse Transformed Data With 3rd Component Removed')

x[:,1] = 0
plot3d(pca.inverse_transform(x), 'PCA Inverse Transformed Data With 2nd and 3rd Components Removed')

