import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dataset_type = "test" #"train"
sortpooling_k = 10


def obtain_dataset(dataset_path):
    df = pd.read_csv(dataset_path, header=None, delimiter=" ")
    print(df.shape)

    random_seed = 42
    dataset_size = len(df)
    indices = np.arange(dataset_size)
    split = dataset_size #50 #10
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    #print(indices)
    #print(indices[:split])
    #print(df.columns)

    dataframe = df[df.columns.tolist()].values[indices[:split]]

    dataframe = pd.DataFrame(dataframe)

    # removes all the columns with only zeros
    dataframe = dataframe.loc[:, (df != 0).any(axis=0)]

    print(dataframe.shape)
    print(dataframe.corr())
    return dataframe


#mask = np.triu(np.ones_like(dataframe.corr(), dtype=bool))

def save_heatmap(correlation_matrix, heatmap_title='Correlation Heatmap', heatmap_filepath='heatmap.png'):
    # Based on https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

    plt.clf()

    # Increase the size of the heatmap.
    plt.figure(figsize=(16, 6))

    # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
    # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
    # (Person is the standard correlation method)
    heatmap = sns.heatmap(correlation_matrix ) #, vmin=0, vmax=1)#, annot=True)

    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title(heatmap_title, fontdict={'fontsize':12}, pad=12)

    # save heatmap as .png file
    # dpi - sets the resolution of the saved image in dots/inches
    # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
    plt.savefig(heatmap_filepath, dpi=300, bbox_inches='tight')


print("========== Non-Vulnerable ========== ")

feat_type = "non-vuln"
dataset_path = "output/{}-features-{}-k{}.csv".format(feat_type, dataset_type, sortpooling_k)
dataframe_nonvuln = obtain_dataset(dataset_path)

print("============ Vulnerable ============ ")

feat_type = "vuln"
dataset_path = "output/{}-features-{}-k{}.csv".format(feat_type, dataset_type, sortpooling_k)
dataframe_vuln = obtain_dataset(dataset_path)

print("==================================== ")

# Verificação se há colunas em um dois dataset que não tem no outro
not_in_non_vulnerable = [i for i in list(dataframe_vuln.columns) if i not in list(dataframe_nonvuln.columns)]
print(not_in_non_vulnerable)
#Todos os itens dos vulneráveis também estão nos não vulneráveis
not_in_vulnerable = [i for i in list(dataframe_nonvuln.columns) if i not in list(dataframe_vuln.columns)]
print(not_in_vulnerable)

for index in not_in_non_vulnerable:
    del(dataframe_vuln[index])

for index in not_in_vulnerable:
    del(dataframe_nonvuln[index])

print("========== Non-Vulnerable ========== ")

feat_type = "non-vuln"
corr_nonvuln = dataframe_nonvuln.corr()
output_heatmap_filepath = 'heatmap-{}-{}-k{}.png'.format(feat_type, dataset_type, sortpooling_k)
save_heatmap(corr_nonvuln, heatmap_filepath=output_heatmap_filepath)

print("============ Vulnerable ============ ")

feat_type = "vuln"
corr_vuln = dataframe_vuln.corr()
output_heatmap_filepath = 'heatmap-{}-{}-k{}.png'.format(feat_type, dataset_type, sortpooling_k)
save_heatmap(corr_vuln, heatmap_filepath=output_heatmap_filepath)

print("==================================== ")

# São 13 linhas (das 19053 linhas) não vulneráveis que tem o valor da feature 329 com um valor diferente de zero

print(corr_nonvuln.shape)
print(corr_vuln.shape)

corr_vuln_np = corr_vuln.to_numpy()
corr_nonvuln_np = corr_nonvuln.to_numpy()

print(corr_nonvuln_np)
print(corr_vuln_np)

diff_numpy = corr_nonvuln_np - corr_vuln_np
df_diff = pd.DataFrame(data=diff_numpy)

print(df_diff)
save_heatmap(df_diff, heatmap_title='Correlation Heatmap Diff {}'.format(dataset_type.capitalize()), heatmap_filepath='heatmap-diff-{}-k{}.png'.format(dataset_type, sortpooling_k))

diff_numpy = np.absolute(diff_numpy)
df_diff = pd.DataFrame(data=diff_numpy)

print(df_diff)
save_heatmap(df_diff, heatmap_title='Correlation Heatmap Diff Absolute {}'.format(dataset_type.capitalize()), heatmap_filepath='heatmap-diff-absolute-{}-k{}.png'.format(dataset_type, sortpooling_k))


