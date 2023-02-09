import torch
from nn import MLP


class Individual:
    def __init__(self, parameters: torch.tensor) -> None:
        self._parameters = parameters
        self._rating = None

    def get_parameters(self) -> torch.tensor:
        return self._parameters

    def get_rating(self) -> float:
        return self._rating

    def set_rating(self, new_rating: float) -> None:
        self._rating = new_rating

    def __lt__(self, other: object) -> bool:
        return self._rating < other.get_rating()

    def __eq__(self, other: object) -> bool:
        return self._rating == other.get_rating()


class EvolutionaryOptimizer:
    def __init__(self, model, criterion, device: str, inheritance_decay: float, mutation_prob: float, mutation_power: float, elite_size: int=1) -> None:
        self._model = model
        self._criterion = criterion
        self._device = device
        self._population = None
        self._best_individual = None

    def evolution(self, data_loader, start_population) -> None:
        self._population = start_population
        self.rate_population(self._population)

        self._best_individual = find_best_individual(self._population)

        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(self._device), labels.to(self._device)

    def eval_individual(self, individual: object, parent: object, data: torch.tensor, labels: torch.tensor) -> None:
        torch.nn.utils.vector_to_parameters(individual.get_parameters(), self._model.parameters())

        output = self._model(data)
        loss = self._criterion(output, labels)

        rating = loss + (parent.get_rating() * (1 - self.inheritance_decay))

        individual.set_rating(rating)





def main():
    model = MLP(3, 2, 1, 4)

    for p in model.parameters():
        p.requires_grad = False


    params = torch.nn.utils.parameters_to_vector(model.parameters())

    params += 1e-3


if __name__ == "__main__":
    main()

