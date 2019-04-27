import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from breast_cancer.helpers import get_std_cancer_data

# code based on towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21

# load breast cancer data-set
data_std, labels = get_std_cancer_data()

# project data onto it's 3 principal components
pca = PCA(n_components=3).fit_transform(data_std)

# calculate variance of each component, then ratio of each variance to total variance
variance = np.var(pca, axis=0)
variance_ratio = variance/np.sum(variance)

print("First two components contribute {}% of the variance".format(int(sum(variance_ratio[:2])*100)))

Xax = pca[:, 0]  # first principal component
Yax = pca[:, 1]  # second principal component

# plotting principal components against each other
cdict = {0: 'red', 1: 'green'}
labl = {0: 'Malignant', 1: 'Benign'}
marker = {0: '*', 1: 'o'}
alpha = {0: .3, 1: .5}
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix = np.where(labels == l)
    ax.scatter(Xax[ix], Yax[ix], c=cdict[l], s=40,
               label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
plt.xlabel("First Principal Component", fontsize=14)
plt.ylabel("Second Principal Component", fontsize=14)
plt.legend()
plt.show()
