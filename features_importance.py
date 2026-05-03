import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preparação dos Dados 
if 'df_train_hibrido' in locals():
    feature_names = df_train_hibrido.drop(columns=['Label']).columns
else:
    feature_names = [f'F{i}' for i in range(1, melhor_modelo.n_features_in_ + 1)]

# Extrair as importâncias do 'melhor_modelo'
importances = melhor_modelo.feature_importances_

# Criar um DataFrame organizando Nome vs. Importância
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Ordenar e pegar as Top 20 para não poluir o gráfico
top_features_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20)

# 2. Plotagem Visual 
plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=top_features_df,
    palette='viridis' 
)

plt.title('Top 20 Features Mais Importantes - Modelo Otimizado (Gemini+Opara)', fontsize=14)
plt.xlabel('Importância (Ganho)', fontsize=12)
plt.ylabel('Feature (ID)', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Adicionar os valores numéricos exatos ao final de cada barra 
for index, value in enumerate(top_features_df['Importance']):
    plt.text(value + 0.001, index, f'{value:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()