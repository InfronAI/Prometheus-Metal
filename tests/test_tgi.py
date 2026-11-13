import pytest

from prometheus_eval.backends import ContextOverflow, GenerationConfig, TGIBackend


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return [1, 2, 3]


@pytest.mark.parametrize(
    "payload", [{"generated_text": "Fine [RESULT] 3"}, [{"generated_text": "Fine [RESULT] 3"}]]
)
def test_tgi_handles_both_original_server_response_shapes(monkeypatch, payload):
    backend = TGIBackend.__new__(TGIBackend)
    backend.server, backend.timeout, backend.tokenizer = (
        "http://test.invalid/generate",
        12,
        Tokenizer(),
    )
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.post", post)
    result = backend.generate("test", GenerationConfig(temperature=0))
    assert result["text"] == "Fine [RESULT] 3"
    assert calls[0][1]["timeout"] == 12
    params = calls[0][1]["json"]["parameters"]
    assert params["do_sample"] is False and "temperature" not in params
    assert calls[0][1]["json"]["inputs"].count("<s>") == 1
    with pytest.raises(ContextOverflow):
        backend.generate("test", GenerationConfig(max_input_tokens=2))
    assert len(calls) == 1
