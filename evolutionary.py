import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn import MLP


class Individual:
    def __init__(self, parameters: torch.tensor) -> None:
        self._parameters = parameters
        self._rating = None

    def get_parameters(self) -> torch.tensor:
        return self._parameters

    def set_parameters(self, new_parameters: torch.tensor) -> None:
        self._parameters = new_parameters

    def get_rating(self) -> float:
        return self._rating

    def set_rating(self, new_rating: float) -> None:
        self._rating = new_rating

    def __lt__(self, other: object) -> bool:
        return self._rating < other.get_rating()

    def __eq__(self, other: object) -> bool:
        return self._rating == other.get_rating()

    def __str__(self) -> str:
        return f"params: {self._parameters}\nrating: {self._rating}"


class EvolutionaryOptimizer:
    def __init__(self, model, criterion, device: str, inheritance_decay: float, mutation_prob: float, mutation_power: float, elite_size: int=1) -> None:
        self._model = model
        self._criterion = criterion
        self._device = device
        self._population = None
        self._best_individual = None
        self.inheritance_decay = inheritance_decay
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self.elite_size = elite_size

    def evolution(self, data_loader, start_population) -> None:
        self._population = start_population
        first_iter = True

        for i, sample in enumerate(data_loader):
            # data, labels = data.to(self._device), labels.to(self._device)
            data = torch.tensor(sample[:-1])
            labels = torch.empty(1, dtype=torch.long).random_(2)[0]

            if (first_iter):
                for individual in self._population:
                    self.rate_individual(individual, data, labels)
                self._best_individual = start_population[0]
                self.find_best_individual()
                first_iter = False

            selected = self.tournament_selection()
            mutants = self.mutate(selected, data, labels)

            self.elitist_succession(mutants)
            self.find_best_individual()
            print(self._best_individual)

    # DO POPRAWY - zrobic jedną funkcję
    def rate_mutant(self, individual: object, parent: object, data: torch.tensor, labels: torch.tensor) -> None:
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(individual.get_parameters(), self._model.parameters())
            output = self._model(data)
            loss = self._criterion(output, labels).item()
            rating = loss + (parent.get_rating() * (1 - self.inheritance_decay))
            individual.set_rating(rating)

    def rate_individual(self, individual, data, labels):
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(individual.get_parameters(), self._model.parameters())
            output = self._model(data)
            loss = self._criterion(output, labels).item()
            individual.set_rating(loss)

    def find_best_individual(self) -> None:
        for individual in self._population:
            if (individual < self._best_individual):
                self._best_individual = individual

    def mutate(self, selected, data, labels):
        mutants = copy.deepcopy(selected)  # DO POPRAWY
        for i, individual in enumerate(mutants):
            if (torch.rand(1).item() < self.mutation_prob):
                old_parameters = individual.get_parameters()
                new_parameters = old_parameters + (self.mutation_power * torch.normal(0, 1, size=(1, len(old_parameters)))[0])
                individual.set_parameters(new_parameters)
                self.rate_mutant(individual, selected[i], data, labels)
        return mutants

    def elitist_succession(self, mutants):
        new_population = sorted(self._population)[:self.elite_size]  # DO POPRAWY
        new_population.extend(sorted(mutants)[:-self.elite_size])  # DO POPRAWY
        self._population = new_population

    def tournament_selection(self):
        # TODO selekcja turniejowa
        return self._population


def main():

    model = MLP(1, 3, 1, 2)
    for p in model.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    device = 'cpu'

    dataset = torch.rand(size=(700, 2))

    start = torch.nn.utils.parameters_to_vector(model.parameters())

    population = []
    for _ in range(20):
        population.append(Individual(start + torch.normal(0, 1, size=(1, len(start)))[0]))


    optimizer = EvolutionaryOptimizer(model, criterion, device, 1e-4, 0.1, 0.8, 1)
    optimizer.evolution(dataset, population)




if __name__ == "__main__":
    main()

