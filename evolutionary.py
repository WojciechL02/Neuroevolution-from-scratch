import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from nn import MLP
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class EvolutionaryOptimizer:
    def __init__(self, model, criterion, device: str, epochs: int, pop_size: int, mutation_prob: float, mutation_power: float) -> None:
        self._model = model
        self._model.eval()
        self._criterion = criterion
        self._device = device
        self._population = None
        self._ratings = None
        self._best_individual_idx = 0
        self._n_params = None
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self._epochs = epochs

    def evolution(self, data_loader, model_params: torch.tensor) -> None:
        self._n_params = model_params.shape[0]
        self._population = torch.empty(size=(self.pop_size, self._n_params), device=self._device)
        self._ratings = torch.empty(size=(self.pop_size,), device=self._device)
        self._population = model_params + torch.normal(0, 1, size=(self.pop_size, self._n_params), device=self._device)

        first_iter = True
        for epoch in range(self._epochs):
            for i, (data, labels) in enumerate(data_loader):
                data, labels = data.to(self._device), labels.to(self._device)

                if (first_iter):
                    for idx, individual in enumerate(self._population):
                        rating = self.rate_individual(individual, data, labels)
                        self._ratings[idx] = rating
                    self.find_best_individual()
                    first_iter = False

                selected, selected_ratings = self.tournament_selection()
                mutants, mutants_ratings, worst_mutant_idx = self.mutate(selected, selected_ratings, data, labels)

            self.elitist_succession(mutants, mutants_ratings, worst_mutant_idx)
            self.find_best_individual()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}:\n{self._ratings[self._best_individual_idx]}")

        torch.nn.utils.vector_to_parameters(self._population[self._best_individual_idx], self._model.parameters())

    def rate_individual(self, individual: torch.tensor, data: torch.tensor, labels: torch.tensor) -> None:
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(individual, self._model.parameters())

            data = data.view(len(data), -1)
            output = self._model(data)
            loss = self._criterion(output, labels.long()).item()
            return loss

    def find_best_individual(self) -> None:
        best_idx = torch.argsort(self._ratings)[0]
        self._best_individual_idx = best_idx

    def mutate(self, selected, selected_ratings, data, labels):
        mutants = selected.detach().clone()
        mutants_ratings = selected_ratings.detach().clone()
        worst_mutant_idx = 0
        for i, individual in enumerate(mutants):
            if (torch.rand(1).item() < self.mutation_prob):
                individual += self.mutation_power * torch.normal(0, 1, size=(self._n_params,), device=self._device)
                mutants_ratings[i] = self.rate_individual(individual, data, labels)
                if (mutants_ratings[i] > mutants_ratings[worst_mutant_idx]):
                    worst_mutant_idx = i
        return mutants, mutants_ratings, worst_mutant_idx

    def elitist_succession(self, mutants, mutants_ratings, worst_mutant_idx):
        self._population[0] = self._population[self._best_individual_idx]
        self._ratings[0] = self._ratings[self._best_individual_idx]
        j = 0
        for i in range(1, self.pop_size):
            if (j != worst_mutant_idx):
                self._population[i] = mutants[j]
                self._ratings[i] = mutants_ratings[j]
            j += 1

    def tournament_selection(self):
        selected = torch.empty(size=self._population.shape)
        selected_ratings = torch.empty(size=self._ratings.shape)
        for i in range(self.pop_size):
            first, second = torch.randint(low=0, high=self.pop_size, size=(2,))
            winner = first if self._ratings[first] < self._ratings[second] else second
            selected[i] = self._population[winner]
            selected_ratings[i] = self._ratings[winner]
        return selected, selected_ratings


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(len(data), -1)
            output = model(data)
            test_loss += criterion(output, target.long())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"\nTest set: Loss: {test_loss}")
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class MyDataset(Dataset):
    def __init__(self, size):
        table = []
        for _ in range(size//2):
            x = round(torch.rand(size=(1,)).item() * 8, 4)
            table.append([x, x**2 - 2*x + 1, 1])
        for _ in range(size//2):
            x = round(torch.rand(size=(1,)).item() * 8, 4)
            table.append([x, 2.71 ** x, 0])
        for _ in range(size//2):
            x = round(torch.rand(size=(1,)).item() * 8, 4)
            table.append([x, 0.9 * x**2 - 3*x + 2, 2])
        df=pd.DataFrame(table, columns=["x", "y", "class"])

        x=df.iloc[:,0:10].values
        y=df.iloc[:,10].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


class KaggleDataset(Dataset):
    def __init__(self, x, y):
        self.x_train=torch.from_numpy(x).float()
        self.y_train=torch.tensor(y.to_numpy())

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


def prepare_dataset():
    df = pd.read_csv("data/Wine_Quality_Data.csv")
    df["color"] = df["color"].map({"red": 1, "white": 0})
    df["quality"] = df["quality"] - 3
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = prepare_dataset()

    train_dataset = KaggleDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    test_dataset = KaggleDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


    model = MLP(12, 7, 1, 4)
    for p in model.parameters():
        p.requires_grad = False
    start_params = torch.nn.utils.parameters_to_vector(model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = EvolutionaryOptimizer(model, criterion, device, 10, 60, 0.52, 0.4)

    optimizer.evolution(train_loader, start_params)
    test(model, device, criterion, test_loader)


if __name__ == "__main__":
    main()

