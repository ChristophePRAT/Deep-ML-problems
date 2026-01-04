from typing import List, Tuple, Union

import torch


def train_neuron(
    features: Union[List[List[float]], torch.Tensor],
    labels: Union[List[float], torch.Tensor],
    initial_weights: Union[List[float], torch.Tensor],
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> Tuple[List[float], float, List[float]]:
    """
    Train a single neuron (sigmoid activation) with mean-squared-error loss.

    Returns (updated_weights, updated_bias, mse_per_epoch)
    â weights & bias are rounded to 4 decimals; each MSE value is rounded too.
    """
    features_tensor = (
        features
        if isinstance(features, torch.Tensor)
        else torch.tensor(features, dtype=torch.float32)
    ).float()
    labels_tensor = (
        labels
        if isinstance(labels, torch.Tensor)
        else torch.tensor(labels, dtype=torch.float32)
    )
    weights_tensor = (
        (
            initial_weights
            if isinstance(initial_weights, torch.Tensor)
            else torch.tensor(initial_weights)
        )
        .clone()
        .detach()
        .float()
        .requires_grad_(True)
    )
    bias_tensor = (
        (
            initial_bias
            if isinstance(initial_bias, torch.Tensor)
            else torch.tensor([initial_bias])
        )
        .clone()
        .detach()
        .float()
        .requires_grad_(True)
    )

    mse_per_epoch = []

    for _ in range(epochs):
        out = features_tensor @ weights_tensor + bias_tensor
        predictions = torch.sigmoid(out).squeeze()

        errors = labels_tensor - predictions

        loss = torch.mean(errors**2)

        mse_per_epoch.append(round(loss.item(), 4))

        loss.backward()

        with torch.no_grad():
            weights_grad = weights_tensor.grad
            bias_grad = bias_tensor.grad

            assert weights_grad is not None
            assert bias_grad is not None

            weights_tensor -= learning_rate * weights_grad
            bias_tensor -= learning_rate * bias_grad

            weights_grad.zero_()
            bias_grad.zero_()

    updated_weights = [round(w.item(), 4) for w in weights_tensor.squeeze()]
    updated_bias = round(bias_tensor.item(), 4)
    return updated_weights, updated_bias, mse_per_epoch


# Tests
def test():
    result_1 = train_neuron(
        torch.tensor([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]),
        torch.tensor([1, 0, 0]),
        torch.tensor([0.1, -0.2]),
        0.0,
        0.1,
        2,
    )

    expected_1 = ([0.1036, -0.1425], -0.0167, [0.3033, 0.2942])

    assert result_1 == expected_1, f"Expected {expected_1}, but got {result_1}"

    result_2 = train_neuron(
        torch.tensor([[1, 2], [2, 3], [3, 1]]),
        torch.tensor([1, 0, 1]),
        torch.tensor([0.5, -0.2]),
        0.0,
        0.1,
        3,
    )
    expected_2 = ([0.4892, -0.2301], 0.0029, [0.21, 0.2087, 0.2076])

    assert result_2 == expected_2, f"Expected {expected_2}, but got {result_2}"

    print("All tests passed.")


if __name__ == "__main__":
    test()
