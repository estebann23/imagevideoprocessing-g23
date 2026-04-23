from imports import X_train, y_train, X_test, test_ids

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

print("training Random forest")
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# crossvalidation accuracy estimation
cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

predictions = model.predict(X_test)

# Save submission
submission = pd.DataFrame({
    "Id": test_ids,
    "Category": predictions,
})

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print("submission.csv saved")
print(submission.head(50))