# ML Lab: All 11 Programs (code only)

## 1. Data Preprocessing (Titanic)

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

show = lambda title, obj: print(f"\n{title}\n{obj}")
data = pd.read_csv("titanic_dataset.csv")
print("First 5 records:", data.head(), sep="\n")
show("Missing values before cleaning:", data.isnull().sum())

show("Missing Age values:", data["Age"].isnull().sum())
age_median = data["Age"].median()
print("Median Age:", age_median)
data["Age"] = data["Age"].fillna(age_median)

show("Missing Embarked values:", data["Embarked"].isnull().sum())
embarked_mode = data["Embarked"].mode()[0]
print("Most frequent Embarked value:", embarked_mode)
data["Embarked"] = data["Embarked"].fillna(embarked_mode)

data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
show("Sex after Label Encoding:", data["Sex"].head())
data = pd.get_dummies(data, columns=["Embarked"], dtype=int)
show("Embarked after One-Hot Encoding:", data.head())

features = ["Age", "SibSp", "Parch", "Fare"]
data[features] = StandardScaler().fit_transform(data[features])

show("Transformed Dataset:", data.head())
show("Missing values after cleaning:", data.isnull().sum())
show("Data types after encoding:", data.dtypes)
show("Scaled numerical features:", data[features].head())
show("Mean of scaled features:", data[features].mean())
show("Standard deviation of scaled features:", data[features].std())

data.to_csv("Titanic_Cleaned.csv", index=False)
print("\nClean dataset saved as: Titanic_Cleaned.csv")
```

## 2. PCA and LDA (Breast Cancer)

```python
import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
print("Original dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Number of classes:", len(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)
print("\nPCA dataset shape:", X_train_pca.shape)

lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_s, y_train)
X_test_lda = lda.transform(X_test_s)
print("LDA dataset shape:", X_train_lda.shape)

def evaluate(X_tr, X_te):
    start = time.time()
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
    elapsed = time.time() - start
    return accuracy_score(y_test, model.predict(X_te)), elapsed

sets = [(X_train_s, X_test_s), (X_train_pca, X_test_pca), (X_train_lda, X_test_lda)]
acc, times = zip(*[evaluate(*s) for s in sets])

results = pd.DataFrame({
    "Method": ["Original (30 Features)", "PCA (10 Components)", "LDA (1 Component)"],
    "Number of Features": [s[0].shape[1] for s in sets],
    "Accuracy": acc,
    "Training Time (seconds)": times})
print("\nRESULTS")
print(results)

print("\nANALYSIS")
for name, a in zip(["Original Features", "PCA", "LDA"], acc):
    print(name, "Accuracy:", round(a * 100, 2), "%")
print()
for name, t in zip(["Original", "PCA", "LDA"], times):
    print(name, "Training Time:", round(t, 6), "seconds")
```

## 3. Linear Regression (California Housing)

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

show = lambda title, obj: print(f"\n{title}\n{obj}")
data = fetch_california_housing(as_frame=True).frame
print("First 5 records:", data.head(), sep="\n")

show("Dataset shape:", data.shape)
show("Missing values:", data.isnull().sum())
data = data.dropna()
show("Missing values after handling:", data.isnull().sum())

X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]
show("Features:", X.columns)
show("Target:", "MedHouseVal")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression().fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
mse = mean_squared_error(y_test, y_pred)

print("\nMODEL PARAMETERS")
print("Intercept:", model.intercept_)
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)
print("\nMODEL EVALUATION")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", np.sqrt(mse))
print("R2 Score:", r2_score(y_test, y_pred))
```

## 4. Polynomial Regression on BMI (Diabetes)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X = X[:, 2].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

degrees = [1, 2, 3, 5, 10, 15]
results = []
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    results.append([degree, train_mse, test_mse, np.sqrt(test_mse),
                    r2_score(y_test, model.predict(X_test))])

results_df = pd.DataFrame(results, columns=[
    "Degree", "Training MSE", "Testing MSE", "Testing RMSE", "Testing R2"])
print(results_df.to_string(index=False))

plt.figure(figsize=(8, 5))
for col in ["Training MSE", "Testing MSE"]:
    plt.plot(degrees, results_df[col], marker="o", label=col)
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("BMI Polynomial Regression: Degree vs MSE")
plt.legend()
plt.grid(True)
plt.show()

best_index = results_df["Testing MSE"].idxmin()
print("Best polynomial degree:", int(results_df.loc[best_index, "Degree"]))
for metric in ["MSE", "RMSE", "R2"]:
    print(f"Best Testing {metric}:", results_df.loc[best_index, f"Testing {metric}"])
```

## 5. Ridge and Lasso (Diabetes)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data["target"] = diabetes.target
print(data.head())
print(data.info())
print(data.isnull().sum())
data = data.dropna()

X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

labels = ["Training MSE", "Testing MSE", "Training RMSE",
          "Testing RMSE", "Training R2", "Testing R2"]
def evaluate(model):
    model.fit(X_train_s, y_train)
    preds = [model.predict(X_train_s), model.predict(X_test_s)]
    mse = [mean_squared_error(t, p) for t, p in zip([y_train, y_test], preds)]
    r2 = [r2_score(t, p) for t, p in zip([y_train, y_test], preds)]
    return mse + list(np.sqrt(mse)) + r2
def report(title, values):
    print(title)
    for label, v in zip(labels, values):
        print(label + ":", v)

linear = LinearRegression()
report("\nLinear Regression", evaluate(linear))

alphas = [0.01, 0.1, 1, 10, 100]
families = {"Ridge": (Ridge, {}), "Lasso": (Lasso, {"max_iter": 10000})}
results = {}
for kind, (Model, kw) in families.items():
    rows = []
    for a in alphas:
        model = Model(alpha=a, **kw)
        values = evaluate(model)
        rows.append([a] + values)
        report(f"\n{kind} Regression Alpha: {a}", values)
        print("Coefficients:", model.coef_)
    results[kind] = pd.DataFrame(rows, columns=["Alpha"] + labels)

for kind, df in results.items():
    print(f"\n{kind} Results")
    print(df)

def show_coefs(model):
    for feature, coef in zip(X.columns, model.coef_):
        print(feature, coef)
print("\nLinear Regression Coefficients")
show_coefs(linear)
best = {}
for kind, (Model, kw) in families.items():
    df = results[kind]
    alpha = df.loc[df["Testing MSE"].idxmin(), "Alpha"]
    best[kind] = Model(alpha=alpha, **kw).fit(X_train_s, y_train)
    print(f"\nBest {kind} Alpha:", alpha)
    print(f"{kind} Coefficients")
    show_coefs(best[kind])

print("\nLasso Zero Coefficients")
for feature, coef in zip(X.columns, best["Lasso"].coef_):
    if np.isclose(coef, 0):
        print(feature)

for metric in ["Testing MSE", "Testing R2"]:
    plt.figure(figsize=(8, 5))
    for kind, df in results.items():
        plt.plot(df["Alpha"], df[metric], marker="o", label=f"{kind} {metric}")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel(metric)
    plt.title(f"Alpha vs {metric}")
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 6. Logistic Regression + Confusion Matrix

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
print(df.head())
print(df.isnull().sum())
df = df.dropna()

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(" " * 17 + "Predicted")
print(" " * 17 + "0     1")
for label, (neg, pos) in zip(["Actual 0", "Actual 1"], cm):
    print(label + "     ", neg, "    ", pos)

print()
for name, value in zip(["TN", "FP", "FN", "TP"], cm.ravel()):
    print(name + ":", value)

print()
for name, metric in [("Precision", precision_score), ("Recall", recall_score),
                     ("F1-score", f1_score)]:
    print(name + ":", metric(y_test, y_pred))
```

## 7. SVM Kernels (Breast Cancer)

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Linear": SVC(kernel="linear", random_state=42),
    "Polynomial": SVC(kernel="poly", degree=3, random_state=42),
    "RBF": SVC(kernel="rbf", random_state=42),
}

rows = []
cms = {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", clf)]).fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    prob = pipe.decision_function(Xte)
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    rows.append({
        "Kernel": name,
        "Accuracy": accuracy_score(yte, yp),
        "Precision": precision_score(yte, yp),
        "Sensitivity": recall_score(yte, yp),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(yte, yp),
        "ROC-AUC": roc_auc_score(yte, prob),
    })
    cms[name] = confusion_matrix(yte, yp)

df = pd.DataFrame(rows).set_index("Kernel").round(4)
print(df)
for name, cm in cms.items():
    print(name, cm.tolist())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["Malignant", "Benign"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.show()
```

## 8. K-Means Clustering (Iris)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

X = load_iris().data
Xs = StandardScaler().fit_transform(X)

ks = list(range(2, 8))
inertia = []
sil = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_))

print(pd.DataFrame({"K": ks, "Inertia": np.round(inertia, 4),
                    "Silhouette Score": np.round(sil, 4)}).to_string(index=False))
best_k = ks[int(np.argmax(sil))]
print("Best K: ", best_k)

final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
print("Clusters: ")
print(final.cluster_centers_.round(4))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(ks, inertia, marker="o")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[1].plot(ks, sil, marker="o", color="green")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Method")
pca = PCA(n_components=2).fit(Xs)
Z = pca.transform(Xs)
C = pca.transform(final.cluster_centers_)
axes[2].scatter(Z[:, 0], Z[:, 1], c=final.labels_, cmap="viridis", s=30)
axes[2].scatter(C[:, 0], C[:, 1], c="red", marker="X", s=200)
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")
axes[2].set_title("K-Means Clusters, K=" + str(best_k))
plt.tight_layout()
plt.show()
```

## 9. DBSCAN Clustering (Wine)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
Xs = StandardScaler().fit_transform(df)

eps = 2.3
min_samples = 5
labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
mask = labels != -1

n_clusters = len(set(labels[mask]))
n_noise = int((~mask).sum())
largest = int(np.bincount(labels[mask]).max()) if n_clusters > 0 else 0
sil = silhouette_score(Xs[mask], labels[mask]) if n_clusters >= 2 else float("nan")

print(pd.DataFrame({
    "Item": ["Number of clusters", "Number of noise points", "Largest cluster size",
             "Silhouette score (non-noise points)"],
    "Result": [n_clusters, n_noise, largest, round(sil, 4)]}).to_string(index=False))

Z = PCA(n_components=2).fit_transform(Xs)
plot_df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1],
                        "Label": np.where(labels == -1, "Noise", "Cluster " + labels.astype(str))})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Label", style=(labels == -1), s=60)
plt.title("DBSCAN (eps=" + str(eps) + ", min_samples=" + str(min_samples) + ")")
plt.show()
```

## 10. Ensemble Methods Comparison

```python
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression (Baseline)": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42))]),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

rows = []
for name, model in models.items():
    t = time.time()
    model.fit(Xtr, ytr)
    elapsed = time.time() - t
    yp = model.predict(Xte)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(yte, yp),
        "Precision (Malignant)": precision_score(yte, yp, pos_label=0),
        "Recall (Malignant)": recall_score(yte, yp, pos_label=0),
        "F1 (Malignant)": f1_score(yte, yp, pos_label=0),
        "Training Time (s)": elapsed,
    })

res = pd.DataFrame(rows).set_index("Model").round(4)
print(res)
print(res["Recall (Malignant)"].idxmax())

metrics = ["Accuracy", "Precision (Malignant)", "Recall (Malignant)", "F1 (Malignant)"]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
res[metrics].plot(kind="bar", ax=axes[0], rot=15)
axes[0].set_ylim(0.8, 1.0)
axes[0].set_ylabel("Score")
sns.barplot(x=res.index, y=res["Training Time (s)"], ax=axes[1])
axes[1].set_ylabel("Seconds")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

for n in [10, 50, 100, 200, 500]:
    t = time.time()
    rf = RandomForestClassifier(n_estimators=n, random_state=42).fit(Xtr, ytr)
    el = time.time() - t
    yp = rf.predict(Xte)
    print(n, round(accuracy_score(yte, yp), 4), round(recall_score(yte, yp, pos_label=0), 4), round(el, 4))
```

## 11. Q-Learning on a 4×4 Grid

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 4
S = N * N
A = 4
start = 0
goal = S - 1
goal_reward = 10.0
step_reward = -1.0
alpha = 0.1
gamma = 0.9
episodes = 1000
max_steps = 100
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
arrows = ["^", "v", "<", ">"]

def step(s, a):
    r, c = divmod(s, N)
    nr, nc = r + moves[a][0], c + moves[a][1]
    if nr < 0 or nr >= N or nc < 0 or nc >= N:
        nr, nc = r, c
    ns = nr * N + nc
    if ns == goal:
        return ns, goal_reward, True
    return ns, step_reward, False

def train(eps, seed=42):
    rng = np.random.default_rng(seed)
    Q = np.zeros((S, A))
    rewards = []
    steps = []
    success = []
    for _ in range(episodes):
        s = start
        total = 0.0
        reached = False
        n = 0
        for n in range(1, max_steps + 1):
            if rng.random() < eps:
                a = int(rng.integers(A))
            else:
                best = np.flatnonzero(Q[s] == Q[s].max())
                a = int(rng.choice(best))
            ns, r, done = step(s, a)
            Q[s, a] += alpha * (r + gamma * Q[ns].max() * (not done) - Q[s, a])
            total += r
            s = ns
            if done:
                reached = True
                break
        rewards.append(total)
        steps.append(n)
        success.append(int(reached))
    return Q, np.array(rewards), np.array(steps), np.array(success)

def policy_grid(Q):
    g = np.full((N, N), "", dtype=object)
    for s in range(S):
        r, c = divmod(s, N)
        g[r, c] = "G" if s == goal else arrows[int(np.argmax(Q[s]))]
    return g
def moving_avg(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")

strategies = {"High exploration (eps=0.5)": 0.5, "Balanced (eps=0.1)": 0.1,
              "Low exploration (eps=0.01)": 0.01}
results = {}
rows = []
for name, eps in strategies.items():
    Q, rew, stp, suc = train(eps)
    results[name] = (Q, rew, suc)
    rows.append({
        "Strategy": name,
        "Avg Reward": rew[-100:].mean(),
        "Success Rate (%)": suc[-100:].mean() * 100,
        "Avg Steps": stp[-100:].mean(),
    })

table = pd.DataFrame(rows).set_index("Strategy").round(3)
print("Results (last 100 episodes):")
print(table)
best = table["Avg Reward"].idxmax()
print("Best strategy:", best)
Qb = results[best][0]
print("Q-table (rows = states 0-15, columns = actions):")
print(pd.DataFrame(Qb, columns=["Up", "Down", "Left", "Right"]).rename_axis("State").round(3))
print("Greedy policy (G = goal, start = top-left):")
print(pd.DataFrame(policy_grid(Qb), index=["Row " + str(i) for i in range(N)],
                   columns=["Col " + str(i) for i in range(N)]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, (Q, rew, suc) in results.items():
    axes[0].plot(moving_avg(rew), label=name)
    axes[1].plot(moving_avg(suc) * 100, label=name)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward (moving average)")
axes[0].legend()
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Success Rate % (moving average)")
axes[1].legend()
plt.tight_layout()
plt.show()
```
