import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def remove_size_one(file_path):
    # Carrega o arquivo original
    df = pd.read_csv(file_path, sep=";", dtype={"size": int})

    # Remove os grafos com tamanho 1
    df_filtered = df[df["size"] > 1]

    # Salva o novo CSV
    output_path = file_path.replace(".csv", "_filtered.csv")
    df_filtered.to_csv(output_path, sep=";", index=False)
    print(f"Arquivo salvo em: {output_path}")
    print(f"Linhas antes: {len(df)}, depois: {len(df_filtered)}")

def dataset_sampler(input_path, output_path, sample_size, ratio=0.2):

    # Lê o CSV
    df = pd.read_csv(input_path, sep=';')

    # Remover grafos com apenas um nó
    #df = df[df['size'] > 1]

    # Garantir tipos corretos
    df['label'] = df['label'].astype(bool)

    # Calcula o número de amostras por classe
    n_true = int(sample_size * ratio)
    n_false = sample_size - n_true

    # Faz sample das classes separadamente
    df_true = df[df['label'] == True].sample(n=n_true, random_state=42)
    df_false = df[df['label'] == False].sample(n=n_false, random_state=42)

    # Junta e embaralha
    df_sampled = pd.concat([df_true, df_false]).sample(frac=1, random_state=42)

    # Força label como bool explícito
    df_sampled['label'] = df_sampled['label'].astype(bool)

    for i, row in df_sampled.iterrows():
        expected_len = row["size"] * 19  # 19 = número de features por nó
        actual_len = len(eval(row["feature_matrix"]))
        if actual_len != expected_len:
            print(f"[‼️ Erro na linha {i}] size={row['size']}, expected_len={expected_len}, actual_len={actual_len}")

    # Escreve para novo ficheiro
    df_sampled.to_csv(output_path, sep=';', index=False)

    print(f"Ficheiro gerado com {sample_size} entradas ({n_true} vulneráveis): {output_path}")


def main():
    #remove_size_one("datasets/cfg-dataset-linux-v0.5.csv")
    dataset_sampler("datasets/cfg-dataset-linux-v0.5.csv", "datasets/cfg-dataset-linux-sample1k.csv", 1000)

if __name__ == "__main__":
    main()