import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn import MLP


class EvolutionaryOptimizer:
    def __init__(self, model, criterion, device: str, pop_size: int, inheritance_decay: float, mutation_prob: float, mutation_power: float) -> None:
        self._model = model
        self._model.eval()
        self._criterion = criterion
        self._device = device
        self._population = None
        self._ratings = None
        self._best_individual_idx = 0
        self._n_params = None
        self.pop_size = pop_size
        self.inheritance_decay = inheritance_decay
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power

    def evolution(self, data_loader, model_params: torch.tensor) -> None:
        self._n_params = model_params.shape[0]
        self._population = torch.empty(size=(self.pop_size, self._n_params), device=self._device)
        self._ratings = torch.empty(size=(self.pop_size,), device=self._device)
        self._population[0] = model_params
        for i in range(1, self.pop_size):
            self._population[i] = model_params + torch.normal(0, 1, size=(self._n_params,), device=self._device)

        first_iter = True

        for i, sample in enumerate(data_loader):
            # data, labels = data.to(self._device), labels.to(self._device)
            data = torch.tensor(sample[:-1])
            labels = torch.empty(1, dtype=torch.long).random_(2)[0]

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
            print(self._population[self._best_individual_idx])
            print(self._ratings[self._best_individual_idx], "\n")

    def rate_individual(self, individual: torch.tensor, data: torch.tensor, labels: torch.tensor, parent_rating: float=None) -> None:
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(individual, self._model.parameters())
            output = self._model(data)
            loss = self._criterion(output, labels).item()
            if (parent_rating):
                return loss + (parent_rating * (1 - self.inheritance_decay))
            return loss

    def find_best_individual(self) -> None:
        for i, rating in enumerate(self._ratings):
            if (rating < self._ratings[self._best_individual_idx]):
                self._best_individual_idx = i

    def mutate(self, selected, selected_ratings, data, labels):
        mutants = selected.detach().clone()
        mutants_ratings = selected_ratings.detach().clone()
        worst_mutant_idx = 0
        for i, individual in enumerate(mutants):
            if (torch.rand(1).item() < self.mutation_prob):
                parent_rating = selected_ratings[i]
                individual += self.mutation_power * torch.normal(0, 1, size=(self._n_params,), device=self._device)
                mutants_ratings[i] = self.rate_individual(individual, data, labels, parent_rating)
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
        # TODO selekcja turniejowa
        return self._population, self._ratings


def main():

    model = MLP(1, 3, 1, 2)
    for p in model.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.rand(size=(700, 2))

    start = torch.nn.utils.parameters_to_vector(model.parameters())

    optimizer = EvolutionaryOptimizer(model, criterion, device, 4, 1e-4, 0.7, 0.2)
    optimizer.evolution(dataset, start)


if __name__ == "__main__":
    main()

