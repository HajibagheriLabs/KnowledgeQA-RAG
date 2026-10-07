import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from knowledgeqa.config import Config, ReaderConfig

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


@dataclass(frozen=True)
class Reader:
    tokenizer: AutoTokenizer
    model: torch.nn.Module
    settings: ReaderConfig
    device: torch.device


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    # best span score minus the no-answer score of the same window; abstain when it is low
    margin: float


def load_reader(config: Config, device: torch.device) -> Reader:
    path = config.paths.model_dir(config.reader.model)
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"{path} has no weights; run `make data` first")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True)
    model = model.to(device).eval()
    if config.reader.fp16 and device.type == "cuda":
        model = model.half()
    return Reader(tokenizer, model, config.reader, device)


@torch.inference_mode()
def read(
    reader: Reader, questions: Sequence[str], passages: Sequence[Sequence[str]]
) -> list[Answer]:
    """Best answer per question over all its passages, by margin over the no-answer score."""
    settings = reader.settings
    pairs = [(i, passage) for i, group in enumerate(passages) for passage in group]
    best = [Answer("", -np.inf) for _ in questions]
    span_limit = torch.ones(settings.max_length, settings.max_length, dtype=torch.bool)
    span_limit = torch.triu(span_limit) & ~torch.triu(
        span_limit, settings.max_answer_tokens
    )
    span_limit = span_limit.to(reader.device)
    for start in range(0, len(pairs), settings.batch_size):
        batch = pairs[start : start + settings.batch_size]
        encoded = reader.tokenizer(
            [questions[i] for i, _ in batch],
            [passage for _, passage in batch],
            truncation="only_second",
            max_length=settings.max_length,
            stride=settings.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        owners = encoded.pop("overflow_to_sample_mapping").tolist()
        offsets = encoded.pop("offset_mapping").tolist()
        is_context = torch.tensor(
            [[sid == 1 for sid in encoded.sequence_ids(f)] for f in range(len(owners))],
            device=reader.device,
        )
        outputs = reader.model(**{k: v.to(reader.device) for k, v in encoded.items()})
        # fp16 logits are too coarse to rank spans, so everything after the model is fp32
        starts, ends = outputs.start_logits.float(), outputs.end_logits.float()
        null = starts[:, 0] + ends[:, 0]
        starts = starts.masked_fill(~is_context, -torch.inf)
        ends = ends.masked_fill(~is_context, -torch.inf)
        spans = (starts[:, :, None] + ends[:, None, :]).masked_fill(
            ~span_limit, -torch.inf
        )
        scores, flat = spans.flatten(1).max(dim=1)
        span_start, span_end = flat // settings.max_length, flat % settings.max_length
        margins = (scores - null).tolist()
        for f, owner in enumerate(owners):
            question, passage = batch[owner]
            if margins[f] > best[question].margin:
                begin = offsets[f][span_start[f].item()][0]
                finish = offsets[f][span_end[f].item()][1]
                best[question] = Answer(passage[begin:finish], margins[f])
    return best


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, squash whitespace, as the official SQuAD script."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    predicted, expected = (
        normalize_answer(prediction).split(),
        normalize_answer(gold).split(),
    )
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not predicted or not expected:
        return float(predicted == expected)
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_answers(
    answers: Mapping[str, Answer], golds: Mapping[str, Sequence[str]], threshold: float
) -> dict[str, float]:
    """EM and F1 on answerable questions, kept apart from abstention on unanswerable ones."""
    em, f1, abstained = [], [], []
    for qid, gold in golds.items():
        predicted = answers[qid].text if answers[qid].margin > threshold else ""
        if gold:
            em.append(max(exact_match(predicted, g) for g in gold))
            f1.append(max(token_f1(predicted, g) for g in gold))
        else:
            abstained.append(float(predicted == ""))
    if not em or not abstained:
        raise ValueError("need both answerable and unanswerable questions to score")
    return {
        "answerable_em": float(np.mean(em)),
        "answerable_f1": float(np.mean(f1)),
        "abstention_accuracy": float(np.mean(abstained)),
        "answerable": len(em),
        "unanswerable": len(abstained),
    }
