import pandas as pd
from pathlib import Path

# download do dataset
DATASET_URL = pd.read_parquet("hf://datasets/celsowm/bbc_news_ptbr/data/train-00000-of-00001-97102f0adab65e78.parquet")

# salvar o dataset em um arquivo local
OUTPUT_PATH = Path("data/raw/bbc_news.parquet")

def main():

    print("Baixando dataset...")
    df = pd.read_parquet(DATASET_URL)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("Dataset baixado com sucesso!")
    print(f"Arquivo: {OUTPUT_PATH}")
    print(f"Quantidade de registros: {len(df):,}")
    print(f"Quantidade de colunas: {len(df.columns)}")
    print(f"Colunas: {list(df.columns)}")

if __name__ == "__main__":
    main()