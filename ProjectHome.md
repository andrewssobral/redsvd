**redsvd** is a library for solving several matrix decompositions including singular value decomposition (SVD), principal component analysis (PCA), and eigen value decomposition. redsvd can handle very large matrix efficiently, and optimized for a truncated SVD of  sparse matrices. For example, redsvd can compute a truncated SVD with top 20 singular values for a 100K x 100K matrix with 1M nonzero entries in less than one second.

The algorithm is based on the randomized algorithm for computing large-scale SVD. Although it uses randomized matrices, the results is very accurate with very high probability. See the experiment part for the detail.

  * [English Page](English.md)

  * [Japanese Page](Japanese.md)