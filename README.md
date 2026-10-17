# Classificação de notícias da BBC em português

Projeto de análise e classificação de notícias em português nas categorias temáticas do dataset `celsowm/bbc_news_ptbr`, disponível no Hugging Face. O fluxo cobre coleta, análise exploratória, deduplicação, limpeza, treinamento de modelos, avaliação e previsão de novas notícias.

> **Escopo atual:** os modelos de classificação preveem uma única categoria por notícia. Notícias associadas a várias categorias são preservadas no dataset processado e podem ser avaliadas separadamente, mas não são usadas no treinamento principal.

## Índice

- [Visão do pipeline](#visão-do-pipeline)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Executar o pipeline de dados](#executar-o-pipeline-de-dados)
- [Análise exploratória](#análise-exploratória)
- [Treinar os modelos](#treinar-os-modelos)
- [Avaliar os modelos](#avaliar-os-modelos)
- [Classificar uma notícia](#classificar-uma-notícia)
- [Dados e resultados observados](#dados-e-resultados-observados)
- [Limitações e reprodução](#limitações-e-reprodução)

## Visão do pipeline

```text
Hugging Face
    |
    v
download_data.py
    |
    v
data/raw/bbc_news_ptbr.parquet
    |                     \
    |                      +--> EDA bruta e diagnóstico de duplicatas/ruídos
    v
 tests/unique.py
    |
    v
data/processed/unique_texts.parquet
    |
    v
data_cleaning.py
    |
    v
data/processed/clean_texts.parquet
    |                         \
    |                          +--> eda3.py
    +--> baseline.py / train.py
                  |
                  v
evaluate.py --> relatórios e matrizes de confusão
                  |
                  v
              predict.py
```

As divisões para treino, validação e teste usam somente notícias com uma categoria. A divisão é estratificada e usa a semente `42`, para manter aproximadamente as proporções das categorias e permitir reproduzir a separação.

## Estrutura do projeto

```text
.
├── download_data.py          # Baixa e salva a base bruta
├── data_cleaning.py          # Limpa os textos e títulos
├── eda.py                    # EDA da base bruta
├── eda2.py                   # EDA da base deduplicada
├── eda3.py                   # EDA da base limpa
├── baseline.py               # TF-IDF + regressão logística
├── train.py                  # Fine-tuning do BERT em português
├── evaluate.py               # Avaliação e comparação dos modelos
├── predict.py                # Previsão para uma notícia informada
├── requirements.txt          # Dependências Python
├── data/
│   ├── raw/                  # Dataset original baixado
│   └── processed/            # Datasets deduplicado e limpo
├── models/                   # Modelos treinados localmente
├── reports/                  # Tabelas, previsões e análises
└── tests/                    # Scripts exploratórios e auxiliares
```

Apesar do nome `tests/`, os scripts ali são principalmente ferramentas de análise e experimentação; não formam uma suíte automatizada de testes.

## Instalação

Recomenda-se Python 3.12, usado no ambiente de desenvolvimento do projeto. Na raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No macOS, o `train.py` usa MPS se o PyTorch o detectar; caso contrário, usa CPU. O código atual não seleciona CUDA explicitamente.

## Executar o pipeline de dados

Execute os comandos a partir da raiz do repositório:

```bash
python download_data.py
python tests/unique.py
python data_cleaning.py
```

### O que cada etapa produz

1. `download_data.py` baixa a partição `train` do dataset Hugging Face e cria `data/raw/bbc_news_ptbr.parquet`. Se o arquivo já existir, o script não baixa novamente.
2. `tests/unique.py` agrupa registros pelo texto exatamente igual. Mantém o primeiro título, data e link do grupo e reúne suas categorias numa lista. Cria `data/processed/unique_texts.parquet`.
3. `data_cleaning.py` normaliza espaços do corpo, remove textos com menos de 200 caracteres e identifica o formato (`texto`, `video` ou `audio`) pelo título original. Remove prefixos de áudio/vídeo e sufixos de duração dos títulos, salvando `data/processed/clean_texts.parquet`.

A deduplicação é por igualdade exata do texto; não normaliza espaços ou pontuação antes de agrupar. A limpeza não faz stemming, lematização, remoção de stopwords nem limpeza geral de HTML ou boilerplate do corpo.

## Análise exploratória

Os scripts podem ser executados individualmente, sempre a partir da raiz:

```bash
python eda.py
python tests/duplicadas.py
python tests/inspect_text.py
python tests/inspect_text2.py
python tests/text_analysis.py
python eda2.py
python eda3.py
```

- `eda.py`: estrutura, nulos e distribuição de categorias da base bruta.
- `tests/duplicadas.py`: linhas duplicadas por texto e conflitos de categoria.
- `tests/inspect_text.py` e `tests/inspect_text2.py`: exemplos de textos curtos/longos e possíveis ruídos, padrões repetidos e espaços.
- `tests/text_analysis.py`: comprimento em caracteres e tokens para apoiar a escolha do limite de entrada do BERT.
- `eda2.py`: qualidade, categorias, combinações, tamanhos e datas da base deduplicada.
- `eda3.py`: categorias, notícias exclusivamente de uma categoria e rótulos por notícia na base limpa.

Os gráficos são salvos em `reports/figures/`.

## Treinar os modelos

### Baseline TF-IDF + regressão logística

```bash
python baseline.py
```

O baseline usa apenas notícias com uma categoria e concatena título e corpo. O TF-IDF representa cada notícia por pesos de palavras; a regressão logística multinomial escolhe uma entre as nove categorias. O pipeline atual usa palavras individuais, ignora termos raros ou quase universais e pondera as classes durante o treino. O modelo treinado é salvo em `models/baseline.joblib`.

**Como interpretar o baseline:** TF-IDF cria um vetor numérico esparso por notícia, com uma posição por termo do vocabulário. Termos frequentes na notícia, mas menos comuns na coleção, tendem a receber pesos maiores. A regressão logística aprende pesos para associar esses vetores às nove classes; com o solver padrão `lbfgs`, a formulação é multinomial, usando softmax para produzir probabilidades que somam 1. É um método relativamente rápido e simples de inspecionar, mas não representa a ordem e o contexto das palavras como um modelo Transformer.

O baseline atual usa unigramas (`ngram_range=(1, 1)`), `C=10`, `class_weight="balanced"`, `min_df=2` e `max_df=0.9`. O ajuste experimental em `tests/tune_baseline.py` encontrou uma configuração diferente como melhor na validação cruzada; ela não foi aplicada automaticamente ao `baseline.py`.

### Ajuste experimental do baseline

```bash
python tests/tune_baseline.py
```

O script testa 12 combinações de hiperparâmetros com validação cruzada estratificada em cinco partes: são 60 ajustes durante a busca, mais um ajuste final com a melhor configuração. A seleção é feita pelo F1 macro médio. Os resultados são salvos em `reports/baseline_tuning_cv.csv`. A busca não substitui os parâmetros em `baseline.py` nem salva automaticamente o melhor modelo; a configuração escolhida precisa ser aplicada manualmente.

### BERT em português

Treino padrão, com limite de 256 tokens:

```bash
python train.py --max-length 256 --saida models/bert
```

Variante com limite de 512 tokens e lote menor para economizar memória:

```bash
python train.py --max-length 512 --batch-size 4 --grad-acc 2 --saida models/bert_512
```

O script parte do modelo pré-treinado `neuralmind/bert-base-portuguese-cased` e faz fine-tuning em nove classes. Treina por três épocas, avalia na validação ao final de cada época e conserva os pesos da época com melhor F1 macro. Notícias acima do limite de tokens são truncadas.

**Como o BERT difere:** o baseline trata cada termo como uma característica independente; o BERT constrói representações contextualizadas, em que a interpretação de uma palavra depende das demais palavras ao redor. No fine-tuning, o modelo pré-treinado e a camada final de classificação são ajustados para as categorias deste projeto. Os logits da camada final são convertidos em probabilidades por softmax, e o modelo escolhe a classe de maior probabilidade.

O limite de 256 ou 512 controla quantos tokens de cada entrada podem ser lidos. O modelo de 512 tokens pode usar mais conteúdo de notícias longas, mas demanda mais memória e tempo. Para manter um lote efetivo de oito exemplos nessa variante, o comando usa `--batch-size 4 --grad-acc 2`.

Para testar o fluxo rapidamente, sem salvar o modelo:

```bash
python train.py --smoke
```

O modo smoke usa apenas 200 exemplos de treino, 100 de validação e uma época; não serve para medir desempenho final. Se a pasta de saída já existir, escolha outro `--saida` ou use `--sobrescrever` conscientemente.

## Avaliar os modelos

Treine o baseline e o BERT antes de avaliar. Avaliação padrão na validação:

```bash
python evaluate.py
```

Avaliar uma variante específica do BERT:

```bash
python evaluate.py --bert-dir models/bert_512
```

Comparar a variante de 512 tokens com outra já avaliada no mesmo conjunto:

```bash
python evaluate.py --bert-dir models/bert_512 --comparar-com models/bert
```

A comparação requer que o relatório de previsões da outra variante já exista. O script verifica que os links e a ordem das notícias correspondem.

A avaliação final no teste exige confirmação explícita:

```bash
python evaluate.py --split teste --abrir-teste
```

Use o conjunto de teste apenas para a avaliação final, depois de encerrar ajustes e decisões de modelo. Para avaliar também as notícias com várias categorias:

```bash
python evaluate.py --ambiguas
```

Nesse modo, uma previsão é considerada compatível quando sua categoria está entre os rótulos verdadeiros da notícia. Isso não é uma avaliação completa de classificação multi-rótulo.

O `evaluate.py` calcula accuracy, F1 macro e métricas por categoria; compara erros dos dois modelos; analisa resultados por formato, ano e tamanho; e gera matriz de confusão, previsões Parquet, CSV de métricas e relatórios Markdown em `reports/`.

### O que significam as métricas

- **Accuracy (acurácia):** proporção de notícias classificadas corretamente. É fácil de entender, mas pode esconder desempenho fraco em categorias menos frequentes.
- **Precisão:** entre as notícias previstas como uma categoria, quantas realmente pertencem a ela. Precisão baixa indica muitos falsos positivos para essa classe.
- **Recall (revocação):** entre as notícias que realmente pertencem a uma categoria, quantas o modelo encontrou. Recall baixo indica muitos falsos negativos.
- **F1:** média harmônica entre precisão e recall; diminui se uma das duas métricas for baixa.
- **F1 macro:** calcula o F1 separadamente para cada categoria e tira a média simples. Cada categoria tem o mesmo peso, independentemente do número de exemplos. Por isso, é a principal métrica usada para escolher a melhor época do BERT e a configuração do GridSearch.
- **Suporte:** número de exemplos reais de cada classe na divisão avaliada.

Na matriz de confusão normalizada por linha, a diagonal corresponde ao recall de cada categoria; as células fora da diagonal mostram para quais classes os exemplos foram confundidos. O relatório também compara previsões pareadas: o teste exato de McNemar considera especificamente os exemplos em que dois modelos discordam e verifica se um tende a acertar mais que o outro.

## Classificar uma notícia

O modelo padrão de previsão é `models/bert_512`. Rode o comando e informe título e texto quando solicitado:

```bash
python predict.py
```

Também é possível passar os argumentos diretamente:

```bash
python predict.py \
  --bert-dir models/bert_512 \
  --titulo "Título da notícia" \
  --arquivo caminho/para/noticia.txt
```

Ou fornecer o texto como argumento:

```bash
python predict.py \
  --bert-dir models/bert_512 \
  --titulo "Título da notícia" \
  --texto "Corpo da notícia com pelo menos 200 caracteres..."
```

O programa limpa espaços e ruídos conhecidos do título, junta título e corpo, aplica o limite de tokens do modelo e mostra as três categorias mais prováveis. Textos menores que 200 caracteres geram um aviso, mas ainda são classificados.

## Dados e resultados observados

Na execução analisada durante o desenvolvimento:

| Etapa | Registros | Significado |
|---|---:|---|
| Base bruta | 8.637 | Registros originais, com uma categoria por linha |
| Após agrupar textos iguais | 6.407 | Textos únicos, preservando categorias distintas como lista |
| Após limpeza | 6.386 | Textos com pelo menos 200 caracteres e títulos tratados |

A base limpa contém nove categorias. Nela, 4.570 notícias têm uma única categoria e 1.816 têm múltiplas categorias. O treino principal utiliza as 4.570 de categoria única.

Os números acima descrevem os arquivos locais analisados e podem mudar se a fonte do dataset for atualizada ou se as regras de processamento forem alteradas.

### Desempenho registrado

As tabelas abaixo foram recalculadas a partir dos arquivos de previsões salvos em `reports/`. A validação contém 686 notícias; o teste também contém 686. Os valores estão em porcentagem.

| Modelo | Divisão | Accuracy | F1 macro |
|---|---|---:|---:|
| Baseline TF-IDF + regressão logística | Validação | 73,3% | 71,9% |
| BERT, limite de 256 tokens | Validação | 76,1% | 74,9% |
| BERT, limite de 512 tokens | Validação | **79,7%** | **78,2%** |
| Baseline TF-IDF + regressão logística | Teste | 71,6% | 70,5% |
| BERT, limite de 512 tokens | Teste | **77,0%** | **75,1%** |

Na validação, o BERT de 512 tokens superou o de 256 em 3,7 pontos percentuais de accuracy e 3,3 pontos de F1 macro. As previsões dos dois BERTs diferiram em 107 das 686 notícias; 55 foram acertadas somente pela versão de 512, contra 30 acertadas somente pela de 256. O teste pareado de McNemar registrou `p = 0,0088`. É evidência favorável à variante de 512 nesta comparação, mas o resultado da validação é exploratório, pois também foi usado para escolher entre modelos.

No teste reservado, o BERT de 512 tokens também ficou acima do baseline nos resultados registrados: 77,0% contra 71,6% de accuracy, e 75,1% contra 70,5% de F1 macro. O teste deve continuar sendo tratado como avaliação final, não como conjunto para ajustar parâmetros.

O ganho agregado não significa que o BERT vence em toda classe. No teste, por exemplo, o F1 da categoria `sociedade` foi 0,349 para o BERT de 512 tokens e 0,417 para o baseline. Consulte `reports/metricas_por_classe_teste_bert_512.csv` para os resultados completos por categoria.

O tuning do baseline registrou F1 macro médio de validação cruzada de 0,714 para sua melhor configuração testada (`(1, 2)` n-gramas, `C=100`, classes balanceadas). Esse número vem de folds dentro do conjunto de treino e não deve ser comparado diretamente ao F1 da validação externa na tabela.

## Limitações e reprodução

- Dados em `data/` e modelos em `models/` são ignorados pelo Git. Um clone novo precisa executar a coleta e o processamento; os modelos precisam ser treinados novamente ou obtidos por outro meio.
- O download requer acesso à internet e disponibilidade do dataset no Hugging Face.
- As dependências estão listadas sem versões fixas. Para reproduzir exatamente um ambiente, será necessário registrar versões testadas.
- A semente `42` torna a divisão reproduzível para a mesma versão e ordem dos dados, mas não garante resultados idênticos entre versões diferentes das bibliotecas ou alterações na fonte.
- Os modelos atuais são de rótulo único. Notícias com várias categorias são preservadas e avaliadas à parte, mas não há ainda um modelo que emita vários rótulos para uma notícia.
- A decisão de limitar o treino a textos com uma categoria muda o escopo: o modelo não aprende diretamente a prever todas as combinações de categorias presentes na base.
