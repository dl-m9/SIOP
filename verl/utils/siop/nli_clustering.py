"""NLI-based semantic clustering for SIOP.

Adapted from InfoReasoner (EntailmentDeberta + get_semantic_ids).
Clusters final answers into semantic outcome modes using bidirectional NLI entailment.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


_GLOBAL_ENTAILMENT_MODEL = None


def get_entailment_model(model_name="microsoft/deberta-v2-xlarge-mnli", device="cpu"):
    """Lazy singleton for the NLI model."""
    global _GLOBAL_ENTAILMENT_MODEL
    if _GLOBAL_ENTAILMENT_MODEL is None or _GLOBAL_ENTAILMENT_MODEL.device != torch.device(device):
        _GLOBAL_ENTAILMENT_MODEL = EntailmentDeberta(model_name, device)
    return _GLOBAL_ENTAILMENT_MODEL


class EntailmentDeberta:
    """DeBERTa-based NLI model for checking bidirectional entailment."""

    def __init__(self, model_name="microsoft/deberta-v2-xlarge-mnli", device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def check_implication(self, text1, text2):
        """Check if text1 entails text2.

        Args:
            text1: str or list[str] (premises)
            text2: str or list[str] (hypotheses)

        Returns:
            (predictions, logits) where predictions are 0=contradiction, 1=neutral, 2=entailment
        """
        inputs = self.tokenizer(
            text1, text2, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
        return predictions.detach(), logits.detach()


def check_entail_batch(text1_list, text2_list, model, entailment_option="bi"):
    """Batch bidirectional entailment check.

    Args:
        text1_list: list of premise strings
        text2_list: list of hypothesis strings
        model: EntailmentDeberta instance
        entailment_option: 'bi' for strict bidirectional, 'loose' for relaxed

    Returns:
        semantically_equivalent: tensor of bools
    """
    pred_1, _ = model.check_implication(text1_list, text2_list)
    pred_2, _ = model.check_implication(text2_list, text1_list)

    if entailment_option == "bi":
        return (pred_1 == 2) & (pred_2 == 2)
    elif entailment_option == "loose":
        no_contradiction = (pred_1 != 0) & (pred_2 != 0)
        not_both_neutral = ~((pred_1 == 1) & (pred_2 == 1))
        return no_contradiction & not_both_neutral
    else:
        raise ValueError(f"Unknown entailment_option: {entailment_option}")


def cluster_answers(answers, question, model, strict=True, chunk_size=64):
    """Cluster answers into semantic outcome modes via NLI.

    Args:
        answers: list of answer strings (K answers for one query)
        question: the query string (prepended for context)
        model: EntailmentDeberta instance
        strict: if True, use bidirectional entailment; else use loose
        chunk_size: batch size for NLI calls

    Returns:
        semantic_ids: list[int], cluster ID for each answer
    """
    if len(answers) == 0:
        return []
    if len(answers) == 1:
        return [0]

    contextualized = [f"{question} {a}" for a in answers]
    semantic_ids = [-1] * len(contextualized)
    next_id = 0
    entail_type = "bi" if strict else "loose"

    for i in range(len(contextualized)):
        if semantic_ids[i] != -1:
            continue
        semantic_ids[i] = next_id
        remaining_indices = [j for j in range(i + 1, len(contextualized)) if semantic_ids[j] == -1]
        if not remaining_indices:
            next_id += 1
            continue

        # Batch check: current string vs all unassigned remaining
        current_list = [contextualized[i]] * len(remaining_indices)
        remaining_list = [contextualized[j] for j in remaining_indices]

        # Process in chunks
        for start in range(0, len(remaining_indices), chunk_size):
            end = min(start + chunk_size, len(remaining_indices))
            equiv = check_entail_batch(
                current_list[start:end], remaining_list[start:end], model, entail_type
            )
            for k, is_equiv in enumerate(equiv):
                if is_equiv:
                    semantic_ids[remaining_indices[start + k]] = next_id
        next_id += 1

    assert -1 not in semantic_ids
    return semantic_ids


def get_cluster_info(semantic_ids, answers):
    """Compute cluster frequencies and select reference answers.

    Args:
        semantic_ids: list[int], cluster ID per answer
        answers: list[str], the original answer strings

    Returns:
        dict: {cluster_id: {"count": int, "frequency": float, "reference": str, "members": list[int]}}
    """
    K = len(semantic_ids)
    clusters = {}
    for i, cid in enumerate(semantic_ids):
        if cid not in clusters:
            clusters[cid] = {"count": 0, "members": [], "reference": answers[i]}
        clusters[cid]["count"] += 1
        clusters[cid]["members"].append(i)

    for cid in clusters:
        clusters[cid]["frequency"] = clusters[cid]["count"] / K

    return clusters
