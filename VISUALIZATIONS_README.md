# 📊 Visualizações de Observabilidade - Federated Learning Personalizado

Este módulo fornece visualizações avançadas para observabilidade do treinamento federado e personalização.

## 🎯 Visualizações Disponíveis

### a) Convergência e Performance
1. **Federated Convergence** (`federated_convergence.png`)
   - Accuracy vs Rounds (modelo global)
   - Loss vs Rounds (modelo global)
   - Linha de tendência para visualizar convergência

2. **Client Accuracy Evolution** (`client_accuracy_evolution.png`)
   - Accuracy de validação por cliente durante fine-tuning
   - Loss de validação por cliente durante fine-tuning

### b) Heterogeneidade dos Clientes
- **Class Distribution** (já gerado automaticamente durante treinamento)
  - Heatmap e bar chart de distribuição de classes por cliente

### c) Contribuição dos Clientes
3. **Client Sample Distribution** (`client_sample_distribution.png`)
   - Número de samples (treino + teste) por cliente
   - Bar chart comparativo

### d) Comparação Global vs Personalizado
4. **Global vs Personalized** (`global_vs_personalized.png`)
   - Comparação de accuracy: modelo global vs modelos personalizados
   - Ganho de performance (Δ accuracy) por cliente

### f) Estrutura do Modelo
5. **Model Parameters Analysis** (`model_parameters_analysis.png`)
   - Frozen vs Trainable parameters por camada
   - Pie chart da distribuição total de parâmetros

### g) Análise de Erros
6. **Confusion Matrix Aggregated** (`confusion_matrix_aggregated.png`)
   - Matriz de confusão global (todos os clientes agregados)
   - Comparação: modelo global vs modelos personalizados

7. **Confusion Matrix Per Client** (`confusion_matrix_per_client.png`)
   - Matriz de confusão individual para cada cliente
   - Comparação lado a lado: global vs personalizado

8. **Error Type Heatmap** (`error_type_heatmap.png`)
   - Heatmap de tipos de erro (TN, FP, FN, TP) por cliente
   - Comparação: modelo global vs modelos personalizados

## 🚀 Como Usar

### Opção 1: Gerar visualizações automaticamente (última run)

```bash
python generate_visualizations.py
```

Isso irá:
- Encontrar automaticamente a última run em `artifacts/`
- Carregar o `results.json` dessa run
- Gerar todos os plots na pasta da run

### Opção 2: Especificar uma run específica

```bash
python generate_visualizations.py --run-dir artifacts/run_005
```

### Opção 3: Customizar caminhos

```bash
python generate_visualizations.py \
    --run-dir artifacts/run_005 \
    --results artifacts/run_005/results.json \
    --save-dir meus_plots
```

## 📝 Uso Programático

Você também pode importar e usar as funções individualmente:

```python
from flower_app.advanced_visualization import (
    plot_federated_convergence,
    plot_client_accuracy_evolution,
    plot_global_vs_personalized,
    plot_confusion_matrices,
    generate_all_visualizations
)

# Gerar todas as visualizações
generate_all_visualizations(
    results_path="artifacts/run_005/results.json",
    run_dir="artifacts/run_005",
    save_dir="artifacts/run_005"
)

# Ou gerar plots individuais
results = load_results_json("results.json")
plot_federated_convergence(results, save_dir="plots")
plot_global_vs_personalized(results, save_dir="plots")
```

## 📂 Estrutura de Saída

Após executar, você terá os seguintes arquivos na pasta especificada:

```
artifacts/run_XXX/
├── results.json
├── federated_convergence.png
├── client_accuracy_evolution.png
├── client_sample_distribution.png
├── global_vs_personalized.png
├── model_parameters_analysis.png
├── confusion_matrix_aggregated.png
├── confusion_matrix_per_client.png
├── error_type_heatmap.png
├── class_distribution_heatmap.png  # gerado durante treinamento
└── class_distribution_bar.png      # gerado durante treinamento
```

## 🎨 Customização

Para customizar as visualizações, edite `/flower_app/advanced_visualization.py`:

- **Cores**: Modifique as paletas de cores nos plots
- **Tamanho**: Ajuste os parâmetros `figsize`
- **Estilo**: Modifique `sns.set_style()` no início do arquivo
- **DPI**: Ajuste `plt.rcParams['figure.dpi']`

## 🔧 Requisitos

Instalado automaticamente via `pyproject.toml`:

```toml
dependencies = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "torch==2.6.0",
]
```

## 💡 Dicas

1. **Após cada treinamento**, execute o gerador de visualizações para análise completa
2. **Compare diferentes runs** gerando plots separados para cada uma
3. **Use W&B** em conjunto para visualizações interativas durante o treinamento
4. **Automatize** adicionando ao final do script de treinamento:

```python
# No final de server_app.py ou após flwr run
from flower_app.advanced_visualization import generate_all_visualizations
generate_all_visualizations(results_path, run_dir, run_dir)
```

## 📊 Métricas Chave a Observar

### Durante Treinamento Federado:
- ✅ Convergência suave da accuracy
- ✅ Redução consistente da loss
- ⚠️ Oscilações podem indicar lr muito alto

### Durante Fine-tuning:
- ✅ Melhoria na accuracy personalizada vs global
- ✅ Redução de overfitting (val_loss não aumenta)
- ⚠️ Degradação em alguns clientes pode indicar dados insuficientes

### Análise de Erros:
- ✅ Redução de FP/FN após personalização
- ✅ Equilíbrio entre precisão e recall
- ⚠️ Muitos FN = modelo conservador (não detecta falhas)
- ⚠️ Muitos FP = modelo agressivo (alarmes falsos)

## 🆘 Troubleshooting

**Erro: "No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**Erro: "No federated rounds data found"**
- Verifique se `results.json` existe e tem a estrutura correta
- Execute o treinamento federado antes de gerar visualizações

**Erro: "No global model found"**
- Certifique-se de que a run contém os checkpoints dos modelos
- Verifique se `run_dir` aponta para o diretório correto

## 📚 Referências

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
