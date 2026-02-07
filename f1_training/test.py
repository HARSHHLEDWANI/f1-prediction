from features.build_features import build_features

X, y = build_features(train_until_season=2022)

print(X.head())
print(y.head())
print("Rows:", len(X))
