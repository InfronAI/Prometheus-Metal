import pytest

mx = pytest.importorskip("mlx.core")

from prometheus_eval.training import completion_loss, completion_mask  # noqa: E402


def test_mask_ignores_prompt_and_padding_but_includes_eos():
    # Tokens: [BOS, prompt, first_feedback, RESULT, score, EOS, pad, pad].
    lengths = mx.array([[2, 6]])
    assert completion_mask(lengths, 7).tolist() == [[False, True, True, True, True, False, False]]


def test_loss_and_gradients_only_come_from_completion():
    import mlx.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = mx.zeros((1, 7, 10))

        def __call__(self, inputs):
            return self.logits

    model = Model()
    (loss, count), grad = nn.value_and_grad(model, completion_loss)(
        model, mx.array([[0, 1, 2, 3, 4, 5, 0, 0]]), mx.array([[2, 6]])
    )
    assert count.item() == 4
    assert loss.item() == pytest.approx(2.302585, rel=1e-5)
    values = grad["logits"].tolist()[0]
    assert all(x == 0 for i in (0, 5, 6) for x in values[i])
    assert any(x != 0 for x in values[1])


def test_ablation_iterator_is_independent_of_sequence_length():
    from itertools import islice
    from prometheus_eval.training import single_example_batches

    full = [([i] * (i + 3), 1) for i in range(1, 5)]
    shortened = [([i, 9, 9], 1) for i in range(1, 5)]
    a = [
        batch[0][0, 0].item() for batch in islice(single_example_batches(full, 1, 20, True, 42), 8)
    ]
    b = [
        batch[0][0, 0].item()
        for batch in islice(single_example_batches(shortened, 1, 20, True, 42), 8)
    ]
    assert a == b
