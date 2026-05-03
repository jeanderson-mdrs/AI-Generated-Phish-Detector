from sklearn.metrics import classification_report

# 1. Preparando os dados isolados
X_gemini_scaled = scaler.transform(gemini_test.drop(columns=['Label']))
X_opara_scaled = scaler.transform(opara_test.drop(columns=['Label']))

# 2. Predições isoladas com o melhor modelo
y_pred_gemini = melhor_modelo.predict(X_gemini_scaled)
y_pred_opara = melhor_modelo.predict(X_opara_scaled)


print("="*60)
print("MÉTRICAS ISOLADAS - CONJUNTO GEMINI (SINTÉTICO)")
print("="*60)
print(classification_report(gemini_test['Label'], y_pred_gemini))

print("\n" + "="*60)
print("MÉTRICAS ISOLADAS - CONJUNTO OPARA (GPT-4o)")
print("="*60)
print(classification_report(opara_test['Label'], y_pred_opara))