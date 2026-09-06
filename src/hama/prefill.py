"""Token log-likelihood scoring with a local Hugging Face causal LM."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class PrefillScorer(Protocol):
    """Score a recorded continuation without generating new tokens."""

    def score(self, prefix: str, continuation: str) -> float: ...


@dataclass(frozen=True)
class PrefillScore:
    """Mean and total target-token log-likelihood."""

    mean_logprob: float
    total_logprob: float
    token_count: int


class QwenPrefillScorer:
    """Prefill scorer backed by a locally loaded Qwen causal language model.

    The model is used only as a likelihood evaluator. Gradients are disabled,
    and log-probabilities are computed only for ``continuation`` tokens.
    ``model_name`` may be either a Hugging Face identifier or a local path.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | dict[str, Any] | None = "auto",
        torch_dtype: str | Any = "auto",
        trust_remote_code: bool = False,
        max_length: int | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as error:
            raise ImportError(
                "Qwen prefill scoring requires the 'attribution' extra: "
                "uv sync --extra attribution"
            ) from error

        self._torch = torch
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if model is None:
            model_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            model_class = (
                AutoModelForImageTextToText
                if model_config.model_type.endswith("_vl")
                else AutoModelForCausalLM
            )
            model = model_class.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        self.model = model
        self.model.eval()
        configured_limit = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        usable_limits = [
            int(value)
            for value in (max_length, configured_limit, tokenizer_limit)
            if isinstance(value, int) and 0 < value < 10**7
        ]
        self.max_length = min(usable_limits) if usable_limits else None

    def score(self, prefix: str, continuation: str) -> float:
        return self.score_detail(prefix, continuation).mean_logprob

    def score_many(
        self,
        items: Sequence[tuple[str, str]],
    ) -> tuple[float, ...]:
        """Score counterfactual prefixes in one padded model forward."""

        return tuple(item.mean_logprob for item in self._score_details(items))

    def score_detail(self, prefix: str, continuation: str) -> PrefillScore:
        return self._score_details(((prefix, continuation),))[0]

    def _score_details(
        self,
        items: Sequence[tuple[str, str]],
    ) -> tuple[PrefillScore, ...]:
        if not items:
            return ()
        encoded: list[tuple[list[int], list[int]]] = []
        for prefix, continuation in items:
            if not prefix or not continuation:
                raise ValueError("prefix and continuation must not be empty")
            prefix_ids = self.tokenizer.encode(
                self._chat_prefix(prefix), add_special_tokens=False
            )
            target_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            if not prefix_ids or not target_ids:
                raise ValueError("prefix and continuation must each contain tokens")
            if self.max_length is not None:
                if len(target_ids) >= self.max_length:
                    raise ValueError("continuation exceeds the model context length")
                prefix_ids = prefix_ids[-(self.max_length - len(target_ids)) :]
            encoded.append((prefix_ids, target_ids))

        max_size = max(len(prefix) + len(target) for prefix, target in encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        offsets: list[int] = []
        for prefix_ids, target_ids in encoded:
            sequence = prefix_ids + target_ids
            offset = max_size - len(sequence)
            offsets.append(offset)
            rows.append([pad_id] * offset + sequence)
            masks.append([0] * offset + [1] * len(sequence))

        device = self._input_device()
        input_ids = self._torch.tensor(rows, dtype=self._torch.long, device=device)
        attention_mask = self._torch.tensor(
            masks, dtype=self._torch.long, device=device
        )
        with self._torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        results: list[PrefillScore] = []
        for row, ((prefix_ids, target_ids), offset) in enumerate(
            zip(encoded, offsets, strict=True)
        ):
            start = offset + len(prefix_ids) - 1
            target_logits = logits[row, start : start + len(target_ids), :]
            log_probs = self._torch.log_softmax(target_logits.float(), dim=-1)
            labels = self._torch.tensor(
                target_ids,
                dtype=self._torch.long,
                device=log_probs.device,
            )
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            total = float(token_log_probs.sum().item())
            results.append(
                PrefillScore(total / len(target_ids), total, len(target_ids))
            )
        return tuple(results)

    def _chat_prefix(self, prefix: str) -> str:
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template) and getattr(self.tokenizer, "chat_template", None):
            return str(
                apply_template(
                    [{"role": "user", "content": prefix}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return prefix

    def _input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._torch.device("cpu")
