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

    def evolution(self, data_loader) -> None:

        scores_history = []
        for epoch in range(self._epochs):
            for i, (data, target) in enumerate(data_loader):
                data, target = data.to(self._device), target.to(self._device)

                self.mi_lambda_weights = []
                self.mi_lambda_biases = []

                output = data
                for i in range(len(self.pop_weights)):
                    l_input_size = self.pop_weights[i].size(1)
                    l_output_size = self.pop_weights[i].size(2)

                    offspring_weights = self._mutate_layers(self.pop_weights[i], l_input_size, l_output_size)
                    offspring_biases = self._mutate_layers(self.pop_biases[i], 1, l_output_size)

                    # CONCATENATION MI+LAMBDA
                    self.mi_lambda_weights.append(torch.cat((self.pop_weights[i], offspring_weights), dim=0))
                    self.mi_lambda_biases.append(torch.cat((self.pop_biases[i], offspring_biases), dim=0))

                    # FORWARD
                    if i < len(self.mi_lambda_weights) - 1:
                        output = F.relu(output @ self.mi_lambda_weights[i] + self.mi_lambda_biases[i])
                    else:
                        output = output @ self.mi_lambda_weights[i] + self.mi_lambda_biases[i]

                # EVALUATION AND SELECTION OF NEW POPULATION
                with torch.no_grad():
                    scores = torch.tensor([F.cross_entropy(model, target) for model in output])
                    top_scores, indices = torch.topk(scores, self._pop_size, largest=False)
                    scores_history.append(top_scores.mean())

                    for i in range(len(self.mi_lambda_weights)):
                        self.pop_weights[i] = self.mi_lambda_weights[i][indices]
                        self.pop_biases[i] = self.mi_lambda_biases[i][indices]
        return scores_history

    def _mutate_layers(self, layers, l_input_size, l_output_size) -> torch.tensor:
        expanded_pop = layers.unsqueeze(1).expand(self._pop_size, 7, l_input_size, l_output_size)
        expanded_pop = expanded_pop.reshape(-1, l_input_size, l_output_size)
        # mutation
        mutation_w = torch.rand_like(expanded_pop)
        expanded_pop += self._mut_pow * mutation_w
        # crossover
        a = torch.rand(7 * self._pop_size, l_input_size, l_output_size)
        permutation = torch.randperm(7 * self._pop_size)
        expanded_pop = a * expanded_pop + (1 - a) * expanded_pop[permutation]
        return expanded_pop

    def get_best_model(self) -> list:
        model = []
        for i in range(len(self.pop_weights)):
            model.append(self.pop_weights[i][0])
            model.append(self.pop_biases[i][0])
        return model
