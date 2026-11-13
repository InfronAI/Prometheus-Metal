import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")


def test_streaming_conversion_loads_and_checks_integrity(tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import load_model

    from prometheus_eval.io import write_json
    from prometheus_eval.models import convert_model

    config = dict(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    model = Model(ModelArgs(**config))
    mx.eval(model.parameters())
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "config.json", config)
    import numpy as np

    weights = {
        key: torch.from_numpy(np.array(value).copy()).float()
        for key, value in tree_flatten(model.parameters())
    }
    torch.save(weights, source / "pytorch_model.bin")
    converted = tmp_path / "converted"
    manifest = convert_model(converted, source)
    assert manifest["bits"] == 4
    restored, _ = load_model(converted)
    ids = mx.array([[1, 2, 3]])
    before, after = model(ids), restored(ids)
    assert before.shape == after.shape == (1, 3, 64)
    assert mx.mean(mx.abs(before - after)).item() < 0.2
    assert convert_model(converted, source) == manifest
    different_source = tmp_path / "different-source"
    different_source.mkdir()
    torch.save({"different": torch.zeros(1)}, different_source / "pytorch_model.bin")
    with pytest.raises(ValueError, match="different source"):
        convert_model(converted, different_source)
    shard = next(converted.glob("*.safetensors"))
    with shard.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        convert_model(converted, source)
