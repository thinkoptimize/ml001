import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score,roc_auc_score

df = pd.read_csv("datasets/selected.csv")
df = df.drop(columns=['Unnamed: 0'])
df.tail(5)

y = df['Label']
X = df.drop(columns=['Label'])

# 1. Adım: Önce Train+Validation ve Test setini ayıralım
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Adım: Train+Validation'ı kendi içinde train ve validation olarak ayıralım
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
# Not: %20 test, geri kalan %80'in %25'i validation oluyor. Yani toplamda:
# Train: %60, Validation: %20, Test: %20


#Model
lgb_model = lgb.LGBMClassifier(is_unbalance=True)

# Hiperparametre arama aralığı
param_grid = {
    'num_leaves': [10, 20,30],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [50, 100],
    'boosting_type': ['gbdt'],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',  # Dengesiz veri seti için accuracy yerine F1 skoru
    verbose=2,
    n_jobs=-1
)

# Modeli eğit
grid_search.fit(X_train, y_train)


print("En İyi Parametreler:", grid_search.best_params_)
#En İyi Parametreler: {'boosting_type': 'gbdt', 'colsample_bytree': 0.7, 'learning_rate': 0.05,
# 'n_estimators': 100, 'num_leaves': 30, 'subsample': 0.7}
# En iyi modeli al ve validation setinde kontrol et
best_model = grid_search.best_estimator_

# Validation seti performansı
y_val_pred = best_model.predict(X_val)
print("\nValidation Set F1 Skoru:", f1_score(y_val, y_val_pred))
#Validation Set F1 Skoru: 0.9923673844418366

print("\nValidation Set Roc_AUC:\n", roc_auc_score(y_val, y_val_pred))
#Validation Set Roc_AUC: 0.9866918472400458


# Test seti performansı
y_test_pred = best_model.predict(X_test)
print("\nTest Set F1 Skoru:", f1_score(y_test, y_test_pred))
#Test Set F1 Skoru: 0.9921327473250428

print("\nTest Set Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
#Test Set Confusion Matrix:
 #[[13627   191]
 #[  670 54290]]


print("\nTest Set Roc AUC:\n", roc_auc_score(y_test, y_test_pred))
# Test Set Roc AUC: 0.986993382784685