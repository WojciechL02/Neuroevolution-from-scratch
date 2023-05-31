import torch
import torch.nn.functional as F


class ESOptimizer:
    def __init__(self, device, model_params, criterion, pop_size, mut_pow, n_epochs) -> None:
        self._model_params = model_params
        self._criterion = criterion
        self._device = device
        self._pop_size = pop_size
        self._mut_pow = mut_pow
        self._epochs = n_epochs
        self._init_population()

    def _init_population(self) -> None:
        self.pop_weights = []
        self.pop_biases = []
        self.input_size = self._model_params["input_size"]
        self.hidden_size = self._model_params["hidden_size"]
        self.output_size = self._model_params["output_size"]
        for i in range(self._model_params["n_hidden"]):
            if i == 0:
                self.pop_weights.append(torch.randn(self._pop_size, self.input_size, self.hidden_size))
            else:
                self.pop_weights.append(torch.randn(self._pop_size, self.hidden_size, self.hidden_size))

            self.pop_biases.append(torch.randn(self._pop_size, 1, self.hidden_size))

        self.pop_weights.append(torch.randn(self._pop_size, self.hidden_size, self.output_size))
        self.pop_biases.append(torch.randn(self._pop_size, 1, self.output_size))

        # self.first_layers = torch.randn(2, 8, 16)
        # self.last_layers = torch.randn(2, 16, 2)
        # self.first_biases = torch.randn(2, 1, 16)
        # self.last_biases = torch.randn(2, 1, 2)

    def evolution(self) -> None:

        for epoch in range(self._epochs):
            # for i, (data, labels) in enumerate(data_loader):
            #     data, labels = data.to(self._device), labels.to(self._device)

            offspring_weights = []
            offspring_biases = []
            mi_lambda_weights = []
            mi_lambda_biases = []

            for i in range(self._model_params["n_hidden"]):
                l_input_size = self.pop_weights[i].size(1)
                l_output_size = self.pop_weights[i].size(2)

                # WEIGHTS
                offspring_weights.append(self.pop_weights[i].unsqueeze(1).expand(self._pop_size, 7, l_input_size, l_output_size))
                offspring_weights[i] = offspring_weights[i].reshape(-1, l_input_size, l_output_size)
                # mutation
                mutation_w = torch.rand_like(offspring_weights[i])
                offspring_weights[i] += self._mut_pow * mutation_w
                # crossover
                a = torch.rand(7 * self._pop_size, l_input_size, l_output_size)
                permutation = torch.randperm(7 * self._pop_size)
                offspring_weights[i] = a * offspring_weights[i] + (1 - a) * offspring_weights[i][permutation]

                # BIASES
                offspring_biases.append(self.pop_biases[i].unsqueeze(1).expand(self._pop_size, 7, 1, l_output_size))
                offspring_biases[i] = offspring_biases[i].reshape(-1, 1, l_output_size)
                # mutation
                mutation_w = torch.rand_like(offspring_biases[i])
                offspring_biases[i] += self._mut_pow * mutation_w
                # crossover
                a = torch.rand(7 * self._pop_size, 1, l_output_size)
                offspring_biases[i] = a * offspring_biases[i] + (1 - a) * offspring_biases[i][permutation]

                # CONCATENATION MI+LAMBDA
                mi_lambda_weights.append(torch.cat((self.pop_weights[i], offspring_weights[i]), dim=0))
                mi_lambda_biases.append(torch.cat((self.pop_biases[i], offspring_biases[i]), dim=0))

            # FORWARD THROUGH ALL MODELS
            data = torch.randn(10, 8)
            target = torch.randint(0, 2, size=(10,))

            output = data
            for i in range(len(offspring_weights)):
                if i < len(offspring_weights) - 1:
                    output = F.relu(output @ offspring_weights[i] + offspring_biases[i])
                else:
                    output = output @ offspring_weights[i] + offspring_biases[i]

            # EVALUATION AND SELECTION OF NEW POPULATION
            scores = torch.tensor([F.cross_entropy(model, target) for model in output])
            _, indices = torch.topk(scores, self._pop_size)
            self.pop_weights = [layer[indices] for layer in offspring_weights]
            self.pop_biases = [layer[indices] for layer in offspring_biases]
        print("AMEN")


def main():
    params = {
        "input_size": 8,
        "output_size": 2,
        "n_hidden": 1,
        "hidden_size": 16,
    }
    optim = ESOptimizer("cpu", params, "criterion", 3, 0.4, 2)

    optim.evolution()


if __name__ == "__main__":
    main()


