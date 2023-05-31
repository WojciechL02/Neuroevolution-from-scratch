import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import CustomDataset
from evolutionary import ESOptimizer
from test import test
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("./data/car_dataset.csv")
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = CustomDataset(X_train, y_train.to_numpy())
    test_dataset = CustomDataset(X_test, y_test.to_numpy())

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model_params = {
        "input_size": 8,
        "output_size": 4,
        "n_hidden": 1,
        "hidden_size": 16,
    }
    EPOCHS = 60
    optimizer = ESOptimizer(device, model_params, "criterion", pop_size=200, mut_pow=0.62, n_epochs=EPOCHS)
    history = optimizer.evolution(train_loader)

    plt.plot(range(len(history)), history)
    plt.title("Score history")
    plt.xlabel("update")
    plt.ylabel("avg score")
    plt.show()

    parameters = optimizer.get_best_model()

    model = nn.Sequential()

    # Iterate over the list and add linear layers to the model
    for i in range(0, len(parameters), 2):
        weight = parameters[i]
        bias = parameters[i + 1]
        layer = nn.Linear(weight.size(0), weight.size(1))
        layer.weight.data = weight.t()
        layer.bias.data = bias
        model.add_module(f"linear_{i // 2}", layer)
        if i < len(parameters) - 2:
            model.add_module(f"relu_{i // 2}", nn.ReLU())

    # TEST
    test(model, device, test_loader)


if __name__ == '__main__':
    main()
