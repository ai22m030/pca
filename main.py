import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
from sklearn import preprocessing

# import the iris dataset:
# 3 target classes: setosa, versicolor, virginica
# 4 predictive features: sepal length, sepal width, petal length, petal width
# 150 samples
beans = pd.read_csv('Dry_Bean.csv')

bt = beans.Class.to_numpy()
le = preprocessing.LabelEncoder()
le.fit(bt)

X = beans.iloc[0:, 0:4].to_numpy()
y = le.transform(bt)
nrComponents = 4
pca = decomposition.PCA(n_components=nrComponents, svd_solver='full')  # set up the pca class
pca.fit(X)  # fit the data
X = pca.transform(X)  # apply dimensionality reduction: X is projected on the principal components

fig, axes = plt.subplots(nrows=nrComponents, ncols=1)
for i in range(0, nrComponents):
    a = axes[i].scatter(X[:, i], X[:, (i + 1) % nrComponents], c=y, cmap=plt.cm.Set1)
    axes[i].set_title("PCA[" + str(i) + "] vs PCA[" + str(i + 1) + "]")
    axes[i].legend(a.legend_elements()[0], le.classes_)

x = np.arange(1, nrComponents + 1)  # 1 to 4 for components
print(pca.explained_variance_ratio_)  # how much the eigenvalues cover
plt.figure()
plt.bar(x, pca.explained_variance_ratio_)
plt.xlabel("principal components")
plt.ylabel("explained variance ratio")

fig, axes = plt.subplots(nrows=nrComponents, ncols=1, figsize=(8, 9),
                         sharey="all", sharex="all")

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
axes[3].set_xticklabels(['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength'])

plt.show()
