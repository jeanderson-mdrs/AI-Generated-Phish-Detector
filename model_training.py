import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


PATH_GEMINI = "/content/drive/MyDrive/Colab Notebooks/Datasets/dataset_b_features.csv"
PATH_OPARA = "/content/drive/MyDrive/Colab Notebooks/Datasets/dataset_opara_traduzido_features.csv"


def preparar_dataset(caminho):
    """Lê o CSV, isola a Label e as 60 features (F1-F60) e limpa NaNs."""
    df = pd.read_csv(caminho)
    # Garante que a primeira coluna seja a Label e as outras comecem com 'F'
    cols = [df.columns[0]] + [c for c in df.columns if c.startswith('F')]
    df = df[cols].copy()
    df.rename(columns={df.columns[0]: 'Label'}, inplace=True)
    # Converte tudo para numérico (Label já deve ser 0/1)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.dropna()

def split_estratificado_manual(df, percent_treino=0.8):
    """Garante que a proporção de Phishing/Legítimo seja idêntica no treino e teste."""
    phishing = df[df['Label'] == 1].sample(frac=1, random_state=42)
    legitimo = df[df['Label'] == 0].sample(frac=1, random_state=42)

    corte_p = int(len(phishing) * percent_treino)
    corte_l = int(len(legitimo) * percent_treino)

    treino = pd.concat([phishing.iloc[:corte_p], legitimo.iloc[:corte_l]])
    teste = pd.concat([phishing.iloc[corte_p:], legitimo.iloc[corte_l:]])

    return treino.sample(frac=1, random_state=42), teste.sample(frac=1, random_state=42)

# --- 1. EXECUÇÃO DO PIPELINE ---

df_gemini = preparar_dataset(PATH_GEMINI)
df_opara = preparar_dataset(PATH_OPARA)

# Realiza Split Estratificado (80/20)")
gemini_train, gemini_test = split_estratificado_manual(df_gemini)
opara_train, opara_test = split_estratificado_manual(df_opara)

# Cria os conjuntos Híbridos 
df_train_hibrido = pd.concat([gemini_train, opara_train]).sample(frac=1, random_state=42)
df_test_hibrido = pd.concat([gemini_test, opara_test]).sample(frac=1, random_state=42)

# --- 2. PREPARAÇÃO PARA O MODELO ---

X_train = df_train_hibrido.drop(columns=['Label'])
y_train = df_train_hibrido['Label']

X_test = df_test_hibrido.drop(columns=['Label'])
y_test = df_test_hibrido['Label']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. TREINAMENTO (XGBOOST) ---

# Treinamento do modelo híbrido 
modelo_hibrido = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    subsample=1.0,
    colsample_bytree=0.8
)

modelo_hibrido.fit(X_train_scaled, y_train)

# --- 4. AVALIAÇÃO FINAL ---

y_pred = modelo_hibrido.predict(X_test_scaled)

print("\n" + "="*50)
print("RESULTADO DA VALIDAÇÃO HÍBRIDA")
print("="*50)
print(classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Real Legítimo', 'Real Phishing'],
            yticklabels=['Pred. Legítimo', 'Pred. Phishing'])
plt.title('Matriz de Confusão: Modelo Híbrido (Gemini + Opara)')
plt.ylabel('Realidade')
plt.xlabel('Predição')
plt.show()


joblib.dump(modelo_hibrido, "/content/drive/MyDrive/Colab Notebooks/Modelos/modelo_hibrido.pkl")
joblib.dump(scaler, "/content/drive/MyDrive/Colab Notebooks/Modelos/scaler_hibrido.pkl")
