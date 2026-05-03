from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 1. Definição do Espaço de Busca
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 2. Configuração da Validação Cruzada (5-Fold Estratificado)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Configuração do Grid Search
grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1', # Foco em otimizar o F1-Score
    verbose=1,
    n_jobs=-1 
)

print("Iniciando Fine-tuning e Cross-validation...")
grid_search.fit(X_train_scaled, y_train)

# 4. Resultados do Melhor Modelo
print(f"\n Melhor combinação encontrada: {grid_search.best_params_}")
print(f" Melhor F1-Score Médio na Cross-Validation: {grid_search.best_score_:.4f}")

# 5. Avaliação Final 
melhor_modelo = grid_search.best_estimator_
y_pred_final = melhor_modelo.predict(X_test_scaled)

print("\n" + "="*50)
print("PERFORMANCE DO MODELO APÓS FINE-TUNING")
print("="*50)
print(classification_report(y_test, y_pred_final))