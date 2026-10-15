from openai import OpenAI
import httpx
import ipaddress
import logging
import time
import concurrent.futures
from urllib.parse import urlparse


def _is_loopback_url(url):
    """Return whether an API URL points at the local machine."""
    if not url:
        return False

    hostname = urlparse(url).hostname
    if not hostname:
        return False
    if hostname.lower() == 'localhost':
        return True

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _create_openai_client(api_key, api_base=None):
    kwargs = {'api_key': api_key}
    if api_base:
        kwargs['base_url'] = api_base
        # On Windows, httpx can inherit the system proxy without honoring the
        # registry's <local> bypass rule.  Keep Ollama traffic off that proxy.
        if _is_loopback_url(api_base):
            kwargs['http_client'] = httpx.Client(trust_env=False)
    return OpenAI(**kwargs)



class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None,
                 api_base=None
                ):
        super().__init__()

        api_key = api_key or 'ollama'
        if api_base is None and not api_key.startswith('sk-') and api_key != 'ollama':
            raise ValueError('OpenAI API key should start with sk-')
        self.client = _create_openai_client(api_key, api_base)
        self.model_path = model_path
        self.request_count = 0
        self.generation_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.post_prompt = None
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=30, target=None):
        if n <= 0:
            raise ValueError('n must be positive')
        if target is None:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            self.system_message = target['pre_prompt']
            self.post_prompt = target['post_prompt']
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
                {"role": "user", "content": self.post_prompt}
            ]
        outputs = []
        failures = 0
        while len(outputs) < n and failures < max_trials:
            try:
                self.request_count += 1
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n - len(outputs),
                )
                usage = getattr(results, 'usage', None)
                if usage is not None:
                    self.prompt_tokens += int(getattr(usage, 'prompt_tokens', 0) or 0)
                    self.completion_tokens += int(getattr(usage, 'completion_tokens', 0) or 0)
                choices = [choice.message.content or '' for choice in results.choices]
                if not choices:
                    raise RuntimeError('model endpoint returned no choices')
                remaining = n - len(outputs)
                outputs.extend(choices[:remaining])
                self.generation_count += min(len(choices), remaining)
            except Exception as e:
                failures += 1
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {failures} / {max_trials} times...")
                if failures < max_trials:
                    time.sleep(failure_sleep_time)

        if len(outputs) != n:
            raise RuntimeError(
                f'model endpoint produced {len(outputs)} of {n} requested completions')
        return outputs

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=30, target=None):
        results = []
        # Match Co-AgentFuzz's default target concurrency and avoid changing
        # latency/GPU pressure merely because a baseline emits a larger batch.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time, target): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class OpenAIEmbeddingLLM():
    def __init__(self, model_path=None, api_key=None, api_base=None):
        api_key = api_key or 'ollama'
        self.client = _create_openai_client(api_key, api_base)
        self.model_path = model_path
    
    def get_embedding(self, prompt):
        response = self.client.embeddings.create(
            input=prompt,
            model=self.model_path
        )
        
        return response.data[0].embedding
