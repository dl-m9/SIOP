"""SIOP token-level reward assignment.

Places pre-computed SIOP process rewards at turn boundary tokens.
Same interface as info_gain.py:compute_score().
"""

import re
import string

from verl.utils.siop.turn_utils import get_assistant_turn_end_positions


def preprocess_text(text):
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_answer(solution_str):
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_f1(prediction, ground_truth):
    """Token-level F1 between prediction and ground truth."""
    if not prediction or not ground_truth:
        return 0.0
    pred_tokens = set(preprocess_text(prediction.lower()).split())
    gt_tokens = set(preprocess_text(ground_truth.lower()).split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction, ground_truth):
    """Exact match after normalization."""
    if not prediction or not ground_truth:
        return 0.0
    return 1.0 if preprocess_text(prediction.lower()) == preprocess_text(ground_truth.lower()) else 0.0


def _char_pos_to_token_idx(char_pos, offset_mapping):
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos < start:
            return max(0, i - 1)
    return len(offset_mapping) - 1


def _find_turn_boundaries_in_decoded(solution_str):
    """Find assistant turn boundaries in decoded text (skip_special_tokens=True).

    With skip_special_tokens=True, <|im_start|> and <|im_end|> are stripped.
    We detect turn boundaries by looking for tool interaction patterns:
      - </tool_call> marks end of an assistant turn that invoked a tool
      - <tool_response> ... </tool_response> marks tool output
      - Text after </tool_response> starts a new assistant turn

    Returns list of char positions where each assistant turn ends.
    """
    turn_ends = []

    # Find all </tool_response> positions — each one marks the boundary
    # between a tool response and the next assistant turn
    tag = "</tool_response>"
    search_pos = 0
    while True:
        pos = solution_str.find(tag, search_pos)
        if pos == -1:
            break
        # The previous assistant turn ends at the </tool_call> before this tool_response
        tc_end = solution_str.rfind("</tool_call>", 0, pos)
        if tc_end != -1:
            turn_ends.append(tc_end + len("</tool_call>"))
        search_pos = pos + len(tag)

    # The final turn ends at the end of the string
    turn_ends.append(len(solution_str))

    return turn_ends


def compute_score(
    solution_str,
    ground_truth,
    data_source,
    siop_process_rewards=None,
    tokenizer=None,
    is_validation=False,
    **kwargs,
):
    """Compute token-level SIOP reward scores.

    Args:
        solution_str: model response string (decoded with skip_special_tokens=True)
        ground_truth: reference answer (for validation metrics only)
        data_source: dataset name
        siop_process_rewards: list of per-turn SIOP rewards (pre-computed)
        tokenizer: HuggingFace tokenizer
        is_validation: if True, return dict with f1/em metrics
        response_token_count: actual number of response tokens (from response_ids),
            used to align scores list length with reward_tensor width

    Returns:
        list[float] of per-token scores, or dict with metrics if is_validation
    """
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")

    response_token_count = kwargs.get("response_token_count", None)
    valid_response_mask = kwargs.get("valid_response_mask", None)

    # Compute validation metrics against ground_truth if available
    extracted = extract_answer(solution_str)
    gt_str = ""
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", [])
        if isinstance(targets, list) and targets:
            gt_str = targets[0]
        elif isinstance(targets, str):
            gt_str = targets
    elif isinstance(ground_truth, str):
        gt_str = ground_truth.split("<|answer_split|>")[0]

    f1_score = compute_f1(extracted, gt_str) if gt_str else 0.0
    em_score = compute_em(extracted, gt_str) if gt_str else 0.0

    # Use response_token_count for scores length if provided (aligns with reward_tensor)
    output_size = response_token_count if response_token_count is not None else 0
    if output_size == 0:
        encoding = tokenizer(solution_str, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        output_size = len(token_ids)
    else:
        token_ids = []
        offset_mapping = []
    tokens_size = len(token_ids)
    scores = [0.0] * output_size

    if output_size == 0:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "scores": scores}
        return scores

    turn_ends: list[int]
    if valid_response_mask is not None:
        turn_ends = get_assistant_turn_end_positions(valid_response_mask[:output_size])
        scale = 1.0
    else:
        encoding = tokenizer(solution_str, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        tokens_size = len(token_ids)
        if tokens_size == 0:
            if is_validation:
                return {"f1": f1_score, "em": em_score, "scores": scores}
            return scores
        # Token index scaling factor: map from re-tokenized indices to actual response indices
        # This handles the mismatch when skip_special_tokens=True removes special tokens
        scale = output_size / tokens_size if tokens_size > 0 else 1.0
        turn_ends = _find_turn_boundaries_in_decoded(solution_str)

    num_turns = len(turn_ends)

    # Debug: log first sample to verify turn detection
    if not hasattr(compute_score, "_debug_logged"):
        compute_score._debug_logged = True
        print(f"[SIOP-reward-debug] tokens_size={tokens_size}, output_size={output_size}, "
              f"scale={scale:.3f}, num_turns={num_turns}, "
              f"turn_ends={turn_ends[:5]}, "
              f"siop_rewards={siop_process_rewards}, f1={f1_score:.4f}")
        # Show first 200 chars of solution_str to verify separator detection
        print(f"[SIOP-reward-debug] solution_str[:200]={solution_str[:200]!r}")

    # If no SIOP rewards, give 0 (unsupervised — no GT dependency).
    if siop_process_rewards is None or len(siop_process_rewards) == 0:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "scores": scores}
        return scores

    # If SIOP rewards are available but only one turn is detected, preserve
    # the terminal SIOP reward instead of degrading to f1-only shaping.
    if num_turns <= 1:
        scores[-1] = siop_process_rewards[-1]
        if is_validation:
            return {"f1": f1_score, "em": em_score, "scores": scores}
        return scores

    # Place SIOP rewards at turn boundary tokens
    num_rewards = len(siop_process_rewards)
    for i in range(min(num_turns, num_rewards)):
        if valid_response_mask is not None:
            mapped_idx = min(turn_ends[i] - 1, output_size - 1)
        else:
            turn_end_char = turn_ends[i]
            if turn_end_char > 0:
                local_idx = _char_pos_to_token_idx(min(turn_end_char - 1, len(solution_str) - 1), offset_mapping)
            else:
                local_idx = 0
            local_idx = min(local_idx, tokens_size - 1)
            mapped_idx = min(int(local_idx * scale), output_size - 1)

        scores[mapped_idx] = siop_process_rewards[i]

    if is_validation:
        return {"f1": f1_score, "em": em_score, "scores": scores}
    return scores
