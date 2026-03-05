import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load mnist
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
X = X / 255.0  # normalize

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# params
C = 10
D = X_train.shape[1]

# priors
counts = np.bincount(y_train, minlength=C)
priors = counts / counts.sum()
log_priors = np.log(priors + 1e-12)

# gaussian params
means = np.zeros((C, D))
vars_ = np.zeros((C, D))

for c in range(C):
    Xc = X_train[y_train == c]
    means[c] = Xc.mean(axis=0)
    vars_[c] = Xc.var(axis=0) + 1e-6

def predict(Xb):
    B = Xb.shape[0]
    scores = np.empty((B, C))

    for c in range(C):
        diff = Xb - means[c]
        log_like = -0.5 * np.log(2 * np.pi * vars_[c]) - 0.5 * (diff ** 2) / vars_[c]
        scores[:, c] = log_like.sum(axis=1) + log_priors[c]

    return np.argmax(scores, axis=1)

# test and print results
y_pred = predict(X_test)
print("Overall accuracy:", accuracy_score(y_test, y_pred))

for c in range(C):
    mask = (y_test == c)
    print("Class", c, "accuracy:", accuracy_score(y_test[mask], y_pred[mask]))