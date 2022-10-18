

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=0.1, edgecolor='w', facecolor='w'))

# colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.jet)
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
x = np.arange(1, nrComponents + 1)  # 1 to 4 for components
print(pca.explained_variance_ratio_)  # how much the eigenvalues cover
plt.figure()
plt.bar(x, pca.explained_variance_ratio_)
plt.xlabel("principal components")
plt.ylabel("explained variance ratio")
fig, axes = plt.subplots(nrows=nrComponents, ncols=1, figsize=(8, 9),
                         sharey=True, sharex=True)
fs = 9
axes[0].bar(x, pca.components_[0])  # loadings for PC1
axes[0].set_title("loadings (components) of PC1", fontsize=fs)
axes[1].bar(x, pca.components_[1])
axes[1].set_title("loadings (components) of PC2", fontsize=fs)
axes[2].bar(x, pca.components_[2])
axes[2].set_title("loadings (components) of PC3", fontsize=fs)
axes[3].bar(x, pca.components_[3])
axes[3].set_title("loadings (components) of PC4", fontsize=fs)
axes[3].set_xticks(x)
axes[3].set_xticklabels(iris.feature_names)


import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
