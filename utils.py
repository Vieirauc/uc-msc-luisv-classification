import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def remove_size_one(file_path):
    df = pd.read_csv(file_path, sep=";", dtype={"size": int})
    df_filtered = df[df["size"] > 1]

    output_path = file_path.replace(".csv", "_filtered.csv")
    df_filtered.to_csv(output_path, sep=";", index=False)
    print(f"Arquivo salvo em: {output_path}")
    print(f"Linhas antes: {len(df)}, depois: {len(df_filtered)}")


def dataset_sampler(input_path, output_path, max_total=5000, ratio=0.2):
    """
    Cria um dataset balanceado ou desbalanceado até um total máximo de amostras.

    :param input_path: Caminho para o CSV original
    :param output_path: Caminho para salvar o CSV amostrado
    :param max_total: Número total máximo de amostras desejado
    :param ratio: Proporção de vulneráveis (entre 0 e 1)
    """
    df = pd.read_csv(input_path, sep=';')
    df['label'] = df['label'].astype(bool)

    # Filtra por classe
    df_true = df[df['label'] == True]
    df_false = df[df['label'] == False]

    # Calcula número máximo possível por classe
    n_true_target = int(max_total * ratio)
    n_false_target = max_total - n_true_target

    n_true = min(n_true_target, len(df_true))
    n_false = min(n_false_target, len(df_false))

    if n_true < n_true_target or n_false < n_false_target:
        print(f"[⚠️] Aviso: Limite de dados. Novo total = {n_true + n_false} (vuln: {n_true}, não-vuln: {n_false})")

    sampled_true = df_true.sample(n=n_true, random_state=42)
    sampled_false = df_false.sample(n=n_false, random_state=42)

    df_sampled = pd.concat([sampled_true, sampled_false]).sample(frac=1, random_state=42)
    df_sampled['label'] = df_sampled['label'].astype(bool)

    # Verificação opcional das features
    for i, row in df_sampled.iterrows():
        expected_len = row["size"] * 19
        actual_len = len(eval(row["feature_matrix"]))
        if actual_len != expected_len:
            print(f"[‼️ Erro na linha {i}] size={row['size']}, expected_len={expected_len}, actual_len={actual_len}")

    df_sampled.to_csv(output_path, sep=';', index=False)
    print(f"Ficheiro gerado: {output_path} com {len(df_sampled)} amostras ({n_true} vulneráveis)")


def main():
    #remove_size_one("datasets/cfg-dataset-linux-v0.5.csv")
    dataset_sampler(
        input_path="datasets/cfg-dataset-linux-v0.5_filtered.csv",
        output_path="datasets/cfg-dataset-linux-v0.5_filtered_1k.csv",
        max_total=1000,
        ratio=0.2
    )


if __name__ == "__main__":
    main()


'''
def main():
    #remove_size_one("datasets/cfg-dataset-linux-v0.5.csv")
    dataset_sampler("datasets/cfg-dataset-linux-v0.5_filtered.csv", "datasets/cfg-dataset-linux-sampleBALANCED.csv", 4320, ratio=0.5)

    # Simula a situação (features - min) / (max / min)
    #features = torch.tensor([5.0, 10.0, 0.0])
    #feat_min = torch.tensor([0.0, 0.0, 0.0])
    #feat_max = torch.tensor([10.0, 0.0, 0.0])  # A 2ª e 3ª colunas têm max == 0

    # Cálculo "errado" da normalização original
    #denominator = feat_max / feat_min  # 10/0 = inf, 0/0 = nan
    #print("Denominator:", denominator)  # Deve mostrar: tensor([inf, nan, nan])

    #numerator = features - feat_min
    #print("Numerator:", numerator)

    #normalized = numerator / denominator
    #print("Normalized result:", normalized)



if __name__ == "__main__":
    main()
    '''