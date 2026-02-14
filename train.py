import argparse
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main(test_size: float, seed: int, C: float) -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocess = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=C)

    X_train_scaled = preprocess.fit_transform(X_train)
    X_test_scaled = preprocess.transform(X_test)

    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)

    print("accuracy:", accuracy_score(y_test, preds))
    print("f1:", f1_score(y_test, preds))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    args = p.parse_args()
    main(args.test_size, args.seed, args.C)
