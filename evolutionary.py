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
        input_size = self._model_params["input_size"]
        hidden_size = self._model_params["hidden_size"]
        output_size = self._model_params["output_size"]
        for i in range(self._model_params["n_hidden"]):
            if i == 0:
                self.pop_weights.append(torch.randn(self._pop_size, input_size, hidden_size))
            else:
                self.pop_weights.append(torch.randn(self._pop_size, hidden_size, hidden_size))

            self.pop_biases.append(torch.randn(self._pop_size, 1, hidden_size))

        self.pop_weights.append(torch.randn(self._pop_size, hidden_size, output_size))
        self.pop_biases.append(torch.randn(self._pop_size, 1, output_size))

        # self.first_layers = torch.randn(2, 8, 16)
        # self.last_layers = torch.randn(2, 16, 2)
        # self.first_biases = torch.randn(2, 1, 16)
        # self.last_biases = torch.randn(2, 1, 2)

    def evolution(self, data_loader) -> None:

        for epoch in range(self._epochs):
            # for i, (data, labels) in enumerate(data_loader):
            #     data, labels = data.to(self._device), labels.to(self._device)


def main():
    params = {
        "input_size": 8,
        "output_size": 2,
        "n_hidden": 1,
        "hidden_size": 16,
    }
    optim = ESOptimizer("cpu", params, 1, 3, 1, 1)
    mi = optim.first_layers.size(0)  # Get the size of mi
    df = torch.randn(10, 8)
    target = torch.randint(0, 2, size=(10,))


    # =========================================
    output = df @ optim.first_layers  # torch.matmul
    output = output + optim.first_biases
    logits = output @ optim.last_layers
    logits = logits + optim.last_biases
    # =========================================


    # =========================================
    # Repeat the tensor mi_osobnikow 7 times along a new dimension
    expanded_mi_osobnikow = optim.first_layers.unsqueeze(1).expand(2, 7, 8, 16)

    # Reshape the tensor to combine the first two dimensions (mi and 7)
    lambda_osobnikow = expanded_mi_osobnikow.reshape(-1, 8, 16)

    # TUTAJ IDZIE MUTACJA I KRZYŻOWANIE
    # MUTACJA
    szum = torch.rand_like(lambda_osobnikow)
    sila_mutacji = 0.1
    lambda_osobnikow = lambda_osobnikow + sila_mutacji * szum
    # KRZYZOWANIE
    a = torch.rand(14, 8, 16)
    permutacja = torch.randperm(7*mi)
    # print(permutacja)
    lambda_osobnikow = a * lambda_osobnikow + (1 - a) * lambda_osobnikow[permutacja]
    # =========================================

    mi_lambda_osobnikow = torch.cat((optim.first_layers, lambda_osobnikow), dim=0)

    # =========================================
    # TUTAJ IDZIE OCENA
    scores = torch.randn(8*mi)

    _, indices = torch.topk(scores, mi)
    nowa_populacja = mi_lambda_osobnikow[indices]
    # print(nowa_populacja.shape)
    # =========================================




if __name__ == "__main__":
    main()
