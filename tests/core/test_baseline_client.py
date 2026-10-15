from types import SimpleNamespace

from PromptFuzz.Fuzzer.gptfuzzer.llm.llm import OpenAILLM


class _Completions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=f"completion-{self.calls}")
        # Some OpenAI-compatible endpoints ignore n and return one choice.
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_generate_tops_up_partial_choice_responses():
    model = OpenAILLM.__new__(OpenAILLM)
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model.model_path = "test-model"
    model.request_count = 0
    model.generation_count = 0
    model.prompt_tokens = 0
    model.completion_tokens = 0
    model.post_prompt = None
    model.system_message = "test"

    outputs = model.generate("seed", n=3, max_trials=1, failure_sleep_time=0)

    assert outputs == ["completion-1", "completion-2", "completion-3"]
    assert model.request_count == 3
    assert model.generation_count == 3
    assert model.prompt_tokens == 30
    assert model.completion_tokens == 12
    assert model.total_tokens == 42
