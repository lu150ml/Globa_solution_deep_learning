# 🌐 Global Solution Deep Learning

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lu150ml/Globa_solution_deep_learning/blob/main/gs_deep_learning.ipynb)

Este repositório documenta a Global Solution de Deep Learning desenvolvida para analisar o impacto da Inteligência Artificial no mercado de trabalho entre 2024 e 2030. O projeto utiliza ciência de dados, aprendizado de máquina e geração de linguagem para mapear riscos, agrupar perfis profissionais e fornecer recomendações personalizadas para o desenvolvimento de carreira.

## 🧠 Visão Geral
- **Notebook principal:** `gs_deep_learning.ipynb`
- **Domínio:** análise de empregabilidade diante da adoção de IA.
- **Objetivo:** identificar grupos de profissionais com comportamentos semelhantes e gerar planos de ação para cada perfil usando um modelo de linguagem hospedado na plataforma Groq.

## 📁 Estrutura do Projeto
| Caminho | Descrição |
| --- | --- |
| `gs_deep_learning.ipynb` | Notebook com todo o fluxo de preparação de dados, modelagem e interface. |
| `README.md` | Documento descritivo do projeto (este arquivo). |

> ℹ️ O notebook foi desenhado para ser executado tanto no Google Colab quanto em ambiente local com Jupyter Notebook/Lab.

## 🗂️ Conjunto de Dados
- **Fonte:** [AI Impact on Job Market (2024-2030) – Kaggle](https://www.kaggle.com/datasets/sahilislam007/ai-impact-on-job-market-20242030).
- **Arquivo esperado:** `ai_job_trends_dataset.csv` (deve estar disponível na mesma pasta do notebook).
- **Variáveis-chave:**
  - Indicadores socioeconômicos (salário mediano, vagas abertas, diversidade etc.).
  - Impacto estimado da IA, risco de automação e formato de trabalho.
  - Status do emprego (variável-alvo utilizada para avaliar o modelo de clusterização).

## 🚀 Pipeline Analítico
1. **📥 Importação de dados** – leitura do CSV e inspeção inicial (formato e colunas).
2. **🧹 Limpeza & análise exploratória** – contagem de valores ausentes e verificação de tipos.
3. **🔤 Codificação categórica** – aplicação de `LabelEncoder` para transformar textos em rótulos numéricos.
4. **📏 Padronização** – normalização de variáveis numéricas com `StandardScaler` para estabilizar o treinamento.
5. **✂️ Split estratificado** – divisão em conjuntos de treino e teste (`train_test_split`) preservando a distribuição da classe alvo.
6. **🧮 Pré-processamento combinado** – uso de `ColumnTransformer` com `OneHotEncoder` para as categóricas e novo `StandardScaler` para as numéricas antes da clusterização.
7. **📊 Clusterização com K-Means** – experimentos com 2 clusters, avaliação por `silhouette_score` e comparação com o status de emprego (accuracy, ARI e NMI).
8. **🧭 Nomeação semântica dos clusters** – mapeamento de cada cluster para rótulos interpretáveis:
   - `0 → Profissões Estáveis / Adaptadas à IA`
   - `1 → Profissões em Risco / Alta Automação`
9. **🧑‍💼 Classificação de novos candidatos** – função `classify_candidate` encapsula pré-processamento, predição do cluster e retorno da classe predominante.
10. **🤖 Geração de recomendações** – integração com a API compatível com OpenAI da Groq (modelo `llama-3.1-8b-instant`) para criar planos de carreira personalizados.
11. **🖥️ Interface interativa** – aplicação Gradio que coleta os atributos do candidato e apresenta, em tempo real, sugestões de requalificação.
12. **📉 Visualizações** – redução de dimensionalidade com PCA para exibir os agrupamentos em 2D.

## 🧩 Principais Componentes do Notebook
- `classify_candidate(candidate_dict, preprocessor, model, cluster_names)` → centraliza a lógica de classificação e interpretação dos clusters.
- Bloco de integração com a **API Groq** (`OpenAI(base_url="https://api.groq.com/openai/v1")`) → gera recomendações textuais estruturadas.
- Função `gerar_recomendacoes_groq(...)` → conecta o formulário Gradio ao modelo de linguagem e retorna o texto exibido na interface.
- Interface `gr.Blocks` com sliders, caixas de texto e botões estilizados para simular perfis profissionais.

## ⚙️ Dependências Principais
| Categoria | Pacotes |
| --- | --- |
| Manipulação de dados | `pandas`, `numpy` |
| Pré-processamento & Modelagem | `scikit-learn` (LabelEncoder, StandardScaler, KMeans, train_test_split, ColumnTransformer, OneHotEncoder, PCA, métricas) |
| Visualização | `matplotlib` |
| Interface | `gradio` |
| IA generativa | `openai` (SDK compatível com Groq) |

> ✅ O Google Colab já inclui a maioria das dependências. Para execução local, utilize um ambiente virtual Python 3.9+.

## 💻 Executando Localmente
1. **Clone o repositório**
   ```bash
   git clone https://github.com/lu150ml/Globa_solution_deep_learning.git
   cd Globa_solution_deep_learning
   ```
2. **Crie e ative um ambiente virtual (opcional, mas recomendado)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt  # Caso crie um arquivo de requisitos
   ```
   Ou instale manualmente:
   ```bash
   pip install pandas numpy scikit-learn gradio openai matplotlib
   ```
4. **Disponibilize o dataset**
   - Faça o download no Kaggle.
   - Posicione o arquivo `ai_job_trends_dataset.csv` na raiz do projeto (mesmo diretório do notebook).
5. **Inicie o Jupyter Notebook/Lab**
   ```bash
   jupyter notebook
   ```
   Abra `gs_deep_learning.ipynb` e execute as células em sequência.

## ☁️ Execução no Google Colab
1. Clique no badge "Abrir no Google Colab" no topo deste README.
2. Faça upload do dataset ou monte o Google Drive contendo o arquivo CSV.
3. Configure a variável de ambiente da API (ver seção abaixo) antes de executar a interface Gradio.

## 🔐 Configuração da Chave da API Groq
A integração com o modelo `llama-3.1-8b-instant` exige uma chave válida da Groq.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
```

1. Crie um arquivo `.env` (opcional) com `GROQ_API_KEY=suachave` ou exporte a variável diretamente no terminal:
   ```bash
   export GROQ_API_KEY="suachave"
   ```
2. Reinicie o kernel/notebook após definir a variável.

## 🖥️ Interface Gradio
- **Título e descrição** orientam o usuário sobre o simulador.
- **Entradas**: campos de texto, números e sliders que representam atributos profissionais (setor, impacto da IA, salários, risco de automação, diversidade etc.).
- **Saída**: caixa de texto expansível exibindo as recomendações geradas pela API Groq.
- **Execução**: a chamada `demo.launch(share=True)` habilita um link público temporário.

## 📈 Métricas e Avaliações
- **Silhouette Score** para verificar a separação dos clusters.
- **Accuracy, Adjusted Rand Index (ARI) e Normalized Mutual Information (NMI)** comparando os clusters com o `Job Status` conhecido.
- **Classification Report** para inspeção das classes predominantes.

Os valores exatos dependem do dataset e dos parâmetros utilizados durante a execução.

## 🧪 Boas Práticas
- Execute o pré-processamento completo antes de avaliar a clusterização.
- Valide o desempenho com diferentes sementes (`random_state`) e número de clusters, caso deseje explorar variações.
- Armazene os objetos treinados (`pre_km`, `km_final`, `cluster_names`) para reutilizar no módulo de recomendações.

## 👥 Equipe
- Luís Henrique Ribeiro – RM559100
- Matheus Henrique Portapilla – RM554481
- Ryan Sales Fernandes – RM558397

## 📬 Suporte
Dúvidas, sugestões ou melhorias? Abra uma issue no repositório ou entre em contato com a equipe.

---

✉️ **Contribuições são bem-vindas!** Faça um fork, crie uma branch e envie um pull request descrevendo suas mudanças.
