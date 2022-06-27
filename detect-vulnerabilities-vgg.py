import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

sortpooling_k = 6
epochs = 500


class VGGnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        #self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])
        self.conv_layers = self.create_conv_layers(VGG_types["VGG11"])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        print(f"x.shape: {x.shape}")
        #x = x.reshape(x.shape[0], 64, int(x.shape[2] / 64), x.shape[3])
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        print(f"x.shape: {x.shape}")
        print(f"x: {x}")
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


def plot(imgs, orig_img, type_transform, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig(f"teste-{type_transform}.png")


# function added by josep
def collate(samples):
    # The input `samples` is a list of pairs
    #  (features, label).
    features, labels = map(list, zip(*samples))
    print(type(features))
    print(type(labels))
    print(f"len(features): {len(features)}")
    print(f"len(labels): {len(labels)}")
    features = [feature_sample.split(" ") for feature_sample in features]
    features = [[float(feature) for feature in feature_sample] for feature_sample in features]
    features = torch.tensor(features)
    print(f"features: {features.shape}")

    pad = (int(1024 * 3.5), int(1024 * 3.5), 0, 0)
    print(pad)
    #features = F.pad(features, pad, "constant", 0) # este é o pad "functional". Talvez o T (transforms) esteja mais correto
    print(f"features: {features.shape}")

    # features tem shape (64, 1024)
    # do que eu entendo, os 64 é deste batch.
    # 1024 é o resultado do batch da DGCNN de 32, que tinham 32 (saída do AMP)
    orig_img = features[0].reshape(32, 32).to('cpu') 
    #orig_img = T.ToPILImage()(features[0].reshape(32, 32).to('cpu')) # O T.Resize só funciona se esta linha for usada
    print(orig_img)

    padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
    #plot(padded_imgs, orig_img, "pad")
    for padded_img in padded_imgs:
        print(type(padded_img), padded_img.shape)

    print(f"features.shape: {features.shape}")
    new_features = features.reshape(features.shape[0], 32, 32)
    padded_imgs = [T.Pad(padding=padding, padding_mode="edge")(new_features) for padding in (3, 10, 30, 50)]
    #plot(padded_imgs, features, "pad-feats")
    for padded_img in padded_imgs:
        print(type(padded_img), padded_img.shape)

    padding_size = int(32 * 15.5)
    padding_size_s = int(32 * 7.5)
    #x1 = F.pad(new_features, (padding_size_s, padding_size_s, padding_size, padding_size))
    x1 = F.pad(new_features, (96, 96, 96, 96))
    print(x1)
    print(f"x1.shape: {x1.shape}")
    print(f"features.shape: {features.shape}")
    features = torch.unsqueeze(x1, 0)

    #resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
    #for resized_img in resized_imgs:
    #    print(resized_img.size)
    #plot(resized_imgs, orig_img, "resize")

    #features = features.reshape(features.shape[0], 2, 64, sortpooling_k)
    #features = features.reshape(features.shape[0], 2, 64, int(features.shape[1] / 128))
    # este 2 é do input_channels da VGGnet
    #features = features.reshape(features.shape[0], 2, int(features.shape[1] / 2), features.shape[2])

    print(type(features)) # este é um tensor
    print(type(labels)) # este é um list
    print(f"shape(features): {features.shape}")
    print(f"shape(labels): {len(labels)}")
    return features, torch.tensor(labels)


def adjust_dataset(dataset):
    for i in range(len(dataset)):
        features = dataset[i, 0]
        features = features.split(" ")
        features = [float(feature_sample) for feature_sample in features]
        features = torch.tensor(features)
        dataset[i, 0] = features
    return dataset



def save_stats(model, stats_dict, artifact_suffix):
    df_stats = pd.DataFrame(stats_dict)
    df_stats.set_index('epoch', inplace=True)
    df_stats['epoch_accuracy'] = df_stats['epoch_accuracy'].astype(np.float64)
    sns.lineplot(data=df_stats)
    plt.savefig('stats/train-results-vgg{}.png'.format(artifact_suffix))

    # %%
    #Evaluate Model!
    model.eval()
    #test_X, test_Y = map(list, zip(*testset))
    test_X, test_Y = map(torch.tensor, zip(*testset)) # the test_X is a list of tensors
    print(test_X)
    #test_X = [float(feature_sample) for feature_sample in test_X]
    #print("AQUI: ", test_X)
    #print("fim AQUI")
    #test_X = torch.tensor(test_X)
    prediction = model(test_X)
    print(torch.argmax(prediction, dim = 1))
    #print(test_Y)
    #print(sum(test_Y))

    # accuracy = torch.mean((test_Y == torch.argmax(prediction, dim = 1)).float())
    # torch.argmax(prediction, dim = 1)
    params = test_Y.detach().numpy(), torch.argmax(prediction, dim = 1).float().detach().numpy()
    report = classification_report(*params, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('stats/classification_report{}.csv'.format(artifact_suffix))
    cm = confusion_matrix(*params, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.savefig('stats/confusion-matrix{}.png'.format(artifact_suffix))
    plt.clf()


if __name__ == "__main__":
    bs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGnet(in_channels=bs).to(device)
    print(model)
    ## N = 3 (Mini batch size)
    #x = torch.randn(1, 3, 224, 224).to(device)
    #print(model(x).shape)


    # Added by josep
    df_vuln = pd.read_csv("output/vuln-features-train-k{}-ep{}-cl_h4.csv".format(sortpooling_k, epochs), header=None)#, delimiter=" ")
    df_vuln["label"] = 1
    print(df_vuln.shape)
    df_non_vuln = pd.read_csv("output/non-vuln-features-train-k{}-ep{}-cl_h4.csv".format(sortpooling_k, epochs), header=None)#, delimiter=" ")
    print(df_non_vuln.shape)
    df_non_vuln["label"] = 0

    # this was previously th df_train
    df = pd.concat([df_vuln, df_non_vuln])

    # %%
    random_seed = 42
    test_split = 0.3
    # Creating data indices for training and validation splits:
    dataset_size = len(df)
    indices = np.arange(dataset_size)
    split = int(np.floor(test_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split : ], indices[:split]

    print(df.columns)
    print(df.columns[0])
    print(df.columns[1])
    df_train = df.iloc[train_indices]
    df_train = df_train.head()
    df_test = df.iloc[test_indices]
    #df_train = df[[]].values[train_indices]
    #df_test = df[[]].values[test_indices]

    #print(df_train)
    print(len(df_train), df_train.shape)
    #indices = np.arange(len(df_train))

    #random_seed = 42
    #np.random.seed(random_seed)
    #np.random.shuffle(indices)

    #df_train = df_train.values[indices]
    df_train = shuffle(df_train).reset_index(drop=True)
    #print(df_train)
    #print(df_train.indices)
    #print(type(df_train))

    trainset = df_train[[0, "label"]].values[:]
    testset = df_test[[0, "label"]].values[:]
    testset = adjust_dataset(testset)

    print(f"type(trainset): {type(trainset)}")
    print(f"trainset.shape: {trainset.shape}")

    print(f"type(testset): {type(testset)}")
    print(f"testset.shape: {testset.shape}")

    # Removi o sampler e troquei o batch_size (era 32)
    data_loader = DataLoader(trainset, batch_size=bs, collate_fn=collate) #, sampler=sampler)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    stats_dict = {
        'epoch': [],
        'epoch_losses' : [],
        'epoch_accuracy' : []
    }

    for epoch in range(1):
        epoch_loss = 0
        #sum_label = 0
        for iter, (bg, label) in enumerate(data_loader): # 32: 32,786 / 64: 56,536
            print(iter, epoch)
            print(f"bg.shape: {bg.shape}")
            #print(f"bg.shape: {bg.squeeze().shape}")
            prediction = model(bg)
            print(f"prediction.shape: {prediction.shape}")
            loss = loss_func(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (iter + 1)
        # print(torch.argmax(prediction, dim = 1))
        accuracy = torch.mean((label == torch.argmax(prediction, dim = 1)).float())
        #print("sum_label:", sum_label)
        #print("count_num_model:", count_num_model)
        print('Epoch {}, loss {:.4f}, acc {:.4f}'.format(epoch, epoch_loss, accuracy))
        stats_dict['epoch'].append(epoch)
        stats_dict['epoch_losses'].append(epoch_loss)
        stats_dict['epoch_accuracy'].append(accuracy)

    save_stats(model, stats_dict, "-first-vgg")


# ref https://github.com/aladdinpersson/Machine-Learning-Collection
