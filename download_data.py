from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "celsowm/bbc_news_ptbr"
OUTPUT_PATH = Path("data/raw/bbc_news_ptbr.parquet")

def download_dataset(dataset_name: str, output_path: Path) -> None:

    if output_path.exists():
        print(f"Dataset já existe em {output_path}. Nada a fazer.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Carrega o dataset e salva em formato Parquet
    dataset = load_dataset(dataset_name)
    df = dataset["train"].to_pandas()
    df.to_parquet(output_path, index=False)
    print(f"Dataset salvo em {output_path}")
    print(f"Linhas: {len(df)} | Colunas: {list(df.columns)}")

if __name__ == "__main__":
    download_dataset(DATASET_NAME, OUTPUT_PATH)
