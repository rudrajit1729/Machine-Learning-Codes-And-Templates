# Principal Component Analysis (PCA)

Principal component analysis (PCA) is an unsupervised statistical procedure that uses an orthogonal transformation 
to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables
called principal components. This transformation is defined in such a way that the first principal component has the largest
possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in 
turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors
(each being a linear combination of the variables and containing n observations)
are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.

Here we use PCA to reduce dimensions of the dataset so that we can fasten the process of preictions and model training
and plot graph of the DV and visuaiise the result.
