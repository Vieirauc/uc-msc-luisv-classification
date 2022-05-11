import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch.optim as optim
from torch.utils.data import DataLoader

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


class VGGnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])
        #self.conv_layers = self.create_conv_layers(VGG_types["VGG11"])

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
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
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

# function added by josep
def collate(samples):
    # The input `samples` is a list of pairs
    #  (features, label).
    features, labels = map(list, zip(*samples))
    features = [feature_sample.split(" ") for feature_sample in features]
    features = [[float(feature) for feature in feature_sample] for feature_sample in features]
    features = torch.tensor(features)
    features = features.reshape(features.shape[0], 2, 64, sortpooling_k)
    return features, torch.tensor(labels)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGnet(in_channels=2).to(device)
    print(model)
    ## N = 3 (Mini batch size)
    #x = torch.randn(1, 3, 224, 224).to(device)
    #print(model(x).shape)


    # Added by josep


    df_vuln = pd.read_csv("output/vuln-features-train-k{}.csv".format(sortpooling_k), header=None)#, delimiter=" ")
    df_vuln["label"] = 1
    print(df_vuln.shape)
    df_non_vuln = pd.read_csv("output/non-vuln-features-train-k{}.csv".format(sortpooling_k), header=None)#, delimiter=" ")
    print(df_non_vuln.shape)
    df_non_vuln["label"] = 0

    df_train = pd.concat([df_vuln, df_non_vuln])
    print(df_train)
    print(len(df_train), df_train.shape)
    #indices = np.arange(len(df_train))

    #random_seed = 42
    #np.random.seed(random_seed)
    #np.random.shuffle(indices)

    #df_train = df_train.values[indices]
    df_train = shuffle(df_train).reset_index(drop=True)
    print(df_train)
    #print(df_train.indices)
    print(type(df_train))

    trainset = df_train[[0, "label"]].values[:]

    # Removi o sampler e troquei o batch_size (era 32)
    data_loader = DataLoader(trainset, batch_size=64, collate_fn=collate) #, sampler=sampler)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        epoch_loss = 0
        #sum_label = 0
        for iter, (bg, label) in enumerate(data_loader):
            print(iter, epoch)
            prediction, h_concat, h_feats = model(bg)
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


# ref https://github.com/aladdinpersson/Machine-Learning-Collection
