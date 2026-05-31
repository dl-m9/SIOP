"""SIOP process reward computation (three-phase pipeline).

Phase 1 — siop_cluster_pass():  Extract answers, NLI clustering, build pseudo-input
           token specs.  CPU only, no model forward pass.
Phase 2 — score_pseudo_inputs():  Send pseudo-inputs to external vLLM scoring server
           via HTTP.  Non-blocking on training GPUs.
Phase 3 — compute_siop_process_rewards():  Consecutive log-prob differences → per-turn
           process rewards.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np

from verl.utils.siop.nli_clustering import get_entailment_model, cluster_answers, get_cluster_info
from verl.utils.siop.scoring_client import get_siop_scoring_client
from verl.utils.siop.turn_utils import get_assistant_turn_ranges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_answer(text):
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _build_pseudo_response_tokens(reference_answer, tokenizer, prefix, suffix):
    """Tokenize PREFIX + reference + SUFFIX and find the reference token range.

    Returns (token_ids, ref_start, ref_end).
    """
    full_text = f"{prefix}{reference_answer}{suffix}"
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    if not token_ids:
        return token_ids, 0, 0

    gt_char_start = len(prefix)
    gt_char_end = len(prefix) + len(reference_answer)

    ref_start = None
    ref_end = None
    for tidx, (cs, ce) in enumerate(offset_mapping):
        if ref_start is None and ce > gt_char_start:
            ref_start = tidx
        if cs < gt_char_end and ce > 0:
            ref_end = tidx + 1

    ref_start = ref_start if ref_start is not None else len(token_ids)
    ref_end = ref_end if ref_end is not None else len(token_ids)
    return token_ids, ref_start, ref_end


# ---------------------------------------------------------------------------
# Phase 1 output
# ---------------------------------------------------------------------------

@dataclass
class SiopClusterResult:
    """Everything needed for Phase 2 (scoring) and Phase 3 (rewards)."""
    bsz: int
    sample_cluster_freq: list       # float per sample
    sample_reference: list          # str per sample (primary/canonical ref; cluster key)

    # One entry per (sample, turn, ref) triple — multi-reference support
    pseudo_prompt_ids: list         # list[list[int]]
    pseudo_response_ids: list       # list[list[int]]
    pseudo_ref_ranges: list         # list[(ref_start, ref_end)]
    sample_turn_map: list           # list[(sample_idx, turn_idx, alpha)]

    # Single-turn samples — terminal reward, no scoring needed
    single_turn_rewards: dict = field(default_factory=dict)

    siop_lambda: float = 0.5
    siop_epsilon: float = 1e-8


# ---------------------------------------------------------------------------
# Phase 1: Clustering + pseudo-input construction (CPU only)
# ---------------------------------------------------------------------------

def siop_cluster_pass(batch, tokenizer, config) -> SiopClusterResult:
    """NLI clustering and pseudo-input token construction.  CPU only."""
    siop_lambda = getattr(config, "siop_lambda", 0.5)
    siop_epsilon = getattr(config, "siop_epsilon", 1e-8)
    nli_model_name = getattr(config, "siop_nli_model", "microsoft/deberta-v2-xlarge-mnli")
    nli_device = getattr(config, "siop_nli_device", "cpu")
    strict = getattr(config, "siop_strict_entailment", True)
    num_refs = int(getattr(config, "siop_num_refs", 1))
    prefix = getattr(config, "siop_reference_prefix",
                     "\nNow there's enough information to answer\n</think>\n<answer>\n")
    suffix = getattr(config, "siop_reference_suffix", "\n</answer><|im_end|>")

    bsz = len(batch.batch["input_ids"])
    index = batch.non_tensor_batch.get("uid", np.arange(bsz, dtype=object))
    prompt_width = batch.batch["prompts"].shape[1]

    # --- Decode responses ---
    response_strs, response_ids_list, response_masks_list = [], [], []
    for i in range(bsz):
        valid_len = int(batch.batch["attention_mask"][i, prompt_width:].sum().item())
        resp_ids = batch.batch["responses"][i][:valid_len]
        response_ids_list.append(resp_ids.tolist())
        response_strs.append(tokenizer.decode(resp_ids, skip_special_tokens=False))
        if "response_mask" in batch.batch.keys():
            response_masks_list.append(batch.batch["response_mask"][i][:valid_len].tolist())
        else:
            response_masks_list.append([1] * valid_len)

    # --- Decode prompts ---
    prompt_ids_list, prompt_strs = [], []
    for i in range(bsz):
        p = batch.batch["prompts"][i]
        p = p[p != tokenizer.pad_token_id]
        prompt_ids_list.append(p.tolist())
        prompt_strs.append(tokenizer.decode(p, skip_special_tokens=False))

    # --- Extract answers ---
    final_answers = [_extract_answer(r) or "" for r in response_strs]

    # --- Group by query ---
    unique_uids, uid_to_indices = [], {}
    for i, uid in enumerate(index):
        k = str(uid)
        if k not in uid_to_indices:
            uid_to_indices[k] = []
            unique_uids.append(k)
        uid_to_indices[k].append(i)

    # --- NLI clustering (remote GPU or local CPU fallback) ---
    sample_cluster_freq = [1.0] * bsz
    sample_reference = [""] * bsz
    # Multi-reference: per sample, list of (ref_text, alpha) with sum alpha = 1
    sample_all_refs: list[list[tuple[str, float]]] = [[] for _ in range(bsz)]

    def _pick_refs(members, valid_answers):
        """Pick up to num_refs unique ref strings from cluster members; uniform α."""
        picked, seen = [], set()
        for m in members:
            ans = valid_answers[m].strip()
            if ans and ans not in seen:
                picked.append(ans)
                seen.add(ans)
                if len(picked) >= num_refs:
                    break
        if not picked:
            return []
        w = 1.0 / len(picked)
        return [(a, w) for a in picked]

    # Build clustering groups for remote server
    cluster_groups = []
    trivial_uids = {}  # uid_key -> (valid_answers, valid_mask, indices)
    for uid_key in unique_uids:
        indices = uid_to_indices[uid_key]
        group_answers = [final_answers[i] for i in indices]
        question = prompt_strs[indices[0]]
        valid_mask = [bool(a) for a in group_answers]
        valid_answers = [a for a, v in zip(group_answers, valid_mask) if v]

        if len(valid_answers) < 2:
            # Trivial: no clustering needed — only assign to samples that have answers
            ref = valid_answers[0] if valid_answers else ""
            for local_idx, global_idx in enumerate(indices):
                if valid_mask[local_idx]:
                    sample_cluster_freq[global_idx] = 1.0
                    sample_reference[global_idx] = ref
                    sample_all_refs[global_idx] = [(ref, 1.0)] if ref else []
                else:
                    sample_cluster_freq[global_idx] = 0.0
                    sample_reference[global_idx] = ""
                    sample_all_refs[global_idx] = []
            continue

        cluster_groups.append({
            "group_id": uid_key,
            "question": question,
            "answers": valid_answers,
            "strict": strict,
        })
        trivial_uids[uid_key] = (valid_answers, valid_mask, indices)

    # Try remote clustering via scoring server
    remote_results = {}
    client = get_siop_scoring_client()
    if client is not None and cluster_groups:
        results = client.cluster_batch(cluster_groups)
        for r in results:
            remote_results[r["group_id"]] = r

    # Apply results (remote or local fallback)
    needs_local = []
    for g in cluster_groups:
        uid_key = g["group_id"]
        valid_answers, valid_mask, indices = trivial_uids[uid_key]

        if uid_key in remote_results:
            r = remote_results[uid_key]
            semantic_ids = r["semantic_ids"]
            # cluster_info keys are strings from JSON
            cluster_info = {int(k): v for k, v in r["cluster_info"].items()}
        else:
            needs_local.append((uid_key, valid_answers, valid_mask, indices, g["question"]))
            continue

        vi = 0
        for local_idx, global_idx in enumerate(indices):
            if valid_mask[local_idx]:
                cid = semantic_ids[vi]
                sample_cluster_freq[global_idx] = cluster_info[cid]["frequency"]
                sample_reference[global_idx] = cluster_info[cid]["reference"]
                sample_all_refs[global_idx] = _pick_refs(cluster_info[cid]["members"], valid_answers)
                vi += 1
            else:
                sample_cluster_freq[global_idx] = 0.0
                sample_reference[global_idx] = ""
                sample_all_refs[global_idx] = []

    # Local CPU fallback for any groups that failed remotely
    if needs_local:
        print(f"[SIOP] Falling back to local NLI for {len(needs_local)} groups", flush=True)
        nli_model = get_entailment_model(nli_model_name, nli_device)
        for uid_key, valid_answers, valid_mask, indices, question in needs_local:
            semantic_ids = cluster_answers(valid_answers, question, nli_model, strict=strict)
            cluster_info = get_cluster_info(semantic_ids, valid_answers)
            vi = 0
            for local_idx, global_idx in enumerate(indices):
                if valid_mask[local_idx]:
                    cid = semantic_ids[vi]
                    sample_cluster_freq[global_idx] = cluster_info[cid]["frequency"]
                    sample_reference[global_idx] = cluster_info[cid]["reference"]
                    sample_all_refs[global_idx] = _pick_refs(cluster_info[cid]["members"], valid_answers)
                    vi += 1
                else:
                    sample_cluster_freq[global_idx] = 0.0
                    sample_reference[global_idx] = ""
                    sample_all_refs[global_idx] = []

    n_ans = sum(1 for a in final_answers if a)
    remote_count = len(cluster_groups) - len(needs_local)
    print(f"[SIOP] Clustering done. {len(unique_uids)} groups "
          f"({remote_count} remote, {len(needs_local)} local), "
          f"avg freq: {np.mean(sample_cluster_freq):.3f}, "
          f"answers: {n_ans}/{bsz}", flush=True)

    # --- Optional reliability calibration: q_θ(c|q) ∝ m(c|q) · exp(η · r(c,q)) ---
    siop_eta = float(getattr(config, "siop_eta", 0.0))
    if siop_eta > 0.0 and client is not None:
        # Collect tool observations per sample
        obs_per_sample = [
            re.findall(r"<tool_response>(.*?)</tool_response>", rs, re.DOTALL)
            for rs in response_strs
        ]
        nli_pairs, pair_sample_idx = [], []
        for i in range(bsz):
            ref_ans = sample_reference[i]
            if not ref_ans:
                continue
            for obs in obs_per_sample[i]:
                obs_t = obs.strip()
                if not obs_t:
                    continue
                nli_pairs.append({"premise": obs_t[:2000], "hypothesis": ref_ans})
                pair_sample_idx.append(i)

        sample_evid_sum = [0.0] * bsz
        sample_evid_cnt = [0] * bsz
        if nli_pairs:
            nli_scores = client.nli_score_batch(nli_pairs)
            for s_idx, score_val in zip(pair_sample_idx, nli_scores):
                sample_evid_sum[s_idx] += float(score_val)
                sample_evid_cnt[s_idx] += 1
        sample_evidence = [
            (sample_evid_sum[i] / sample_evid_cnt[i]) if sample_evid_cnt[i] > 0 else 0.0
            for i in range(bsz)
        ]

        # Per query: aggregate per-cluster evidence and re-normalize q_κ
        for uid_key in unique_uids:
            indices = uid_to_indices[uid_key]
            buckets: dict[str, list[int]] = {}
            for gi in indices:
                ref_str = sample_reference[gi]
                if ref_str:
                    buckets.setdefault(ref_str, []).append(gi)
            if len(buckets) <= 1:
                continue
            K_valid = sum(len(v) for v in buckets.values())
            logits = {}
            for ref_str, idxs in buckets.items():
                m_c = len(idxs) / max(K_valid, 1)
                evids = [sample_evidence[i] for i in idxs if sample_evid_cnt[i] > 0]
                r_c = float(np.mean(evids)) if evids else 0.0
                logits[ref_str] = math.log(m_c + 1e-12) + siop_eta * r_c
            max_l = max(logits.values())
            exps = {r: math.exp(v - max_l) for r, v in logits.items()}
            Z = sum(exps.values())
            q_k = {r: v / Z for r, v in exps.items()}
            for ref_str, idxs in buckets.items():
                for gi in idxs:
                    sample_cluster_freq[gi] = q_k[ref_str]

        print(f"[SIOP] Reliability calibration applied (η={siop_eta}, "
              f"{len(nli_pairs)} NLI pairs, "
              f"avg evidence={float(np.mean(sample_evidence)) if sample_evidence else 0:.3f}, "
              f"avg q_θ={float(np.mean(sample_cluster_freq)):.3f})", flush=True)

    # Debug
    if bsz > 0:
        turns0 = get_assistant_turn_ranges(response_masks_list[0])
        print(f"[SIOP] First sample: answer='{final_answers[0]}', turns={len(turns0)}", flush=True)
        for dbg_i, resp in enumerate(response_strs):
            if "<tool_call>" in resp and "</tool_response>" in resp:
                mt = get_assistant_turn_ranges(response_masks_list[dbg_i])
                print(f"[SIOP] Multi-turn sample idx={dbg_i}: turns={len(mt)}, "
                      f"tool_calls={resp.count('</tool_call>')}", flush=True)
                break

    # --- IGPO mode: override cluster ref with gold y* (verifier-supervised) ---
    use_gold_ref = bool(getattr(config, "siop_use_gold_ref", False))
    if use_gold_ref:
        reward_model_field = batch.non_tensor_batch.get("reward_model", None)
        if reward_model_field is None:
            raise ValueError("siop_use_gold_ref=True requires 'reward_model' in non_tensor_batch")
        n_with_gold = 0
        for i in range(bsz):
            rm = reward_model_field[i] if isinstance(reward_model_field[i], dict) else {}
            gt = rm.get("ground_truth", "")
            if isinstance(gt, dict):
                targets = gt.get("target", [])
            elif isinstance(gt, str):
                targets = gt.split("<|answer_split|>") if gt else []
            else:
                targets = []
            gold = targets[0].strip() if targets and isinstance(targets[0], str) else ""
            if gold:
                sample_reference[i] = gold
                sample_cluster_freq[i] = 1.0
                sample_all_refs[i] = [(gold, 1.0)]
                n_with_gold += 1
            else:
                sample_cluster_freq[i] = 0.0
                sample_reference[i] = ""
                sample_all_refs[i] = []
        print(f"[IGPO] gold-ref override: {n_with_gold}/{bsz} samples have gold", flush=True)

    # --- Build pseudo-input token specs (multi-reference) ---
    ref_token_data = {}
    for refs in sample_all_refs:
        for ref_ans, _ in refs:
            if ref_ans and ref_ans not in ref_token_data:
                ref_token_data[ref_ans] = _build_pseudo_response_tokens(ref_ans, tokenizer, prefix, suffix)

    all_prompt_ids, all_response_ids, all_ref_ranges, sample_turn_map = [], [], [], []
    single_turn_rewards = {}

    for i in range(bsz):
        refs_i = sample_all_refs[i]
        if not refs_i or sample_cluster_freq[i] == 0.0:
            continue

        # Drop refs whose tokenization has empty ref range (safety)
        valid_refs = []
        for ref_ans, alpha in refs_i:
            rt, rs, re_ = ref_token_data.get(ref_ans, ([], 0, 0))
            if rs < re_:
                valid_refs.append((ref_ans, alpha, rt, rs, re_))
        if not valid_refs:
            continue
        # Renormalize alphas in case some refs were dropped
        alpha_sum = sum(a for _, a, *_ in valid_refs)
        if alpha_sum <= 0:
            continue
        valid_refs = [(ra, a / alpha_sum, rt, rs, re_) for (ra, a, rt, rs, re_) in valid_refs]

        turns = get_assistant_turn_ranges(response_masks_list[i])
        if len(turns) <= 1:
            single_turn_rewards[i] = [(1.0 - siop_lambda) * sample_cluster_freq[i]]
            continue

        # τ_0 = prompt-only baseline (no response tokens) — one entry per ref
        for _, alpha, rt, rs, re_ in valid_refs:
            all_prompt_ids.append(prompt_ids_list[i])
            all_response_ids.append(rt)
            all_ref_ranges.append((rs, re_))
            sample_turn_map.append((i, -1, alpha))

        # τ_1 ... τ_T = after each assistant turn — one entry per ref per turn
        for t_idx, (_, turn_end) in enumerate(turns):
            turn_end = min(turn_end, len(response_ids_list[i]))
            context_ids = prompt_ids_list[i] + response_ids_list[i][:turn_end]
            for _, alpha, rt, rs, re_ in valid_refs:
                all_prompt_ids.append(context_ids)
                all_response_ids.append(rt)
                all_ref_ranges.append((rs, re_))
                sample_turn_map.append((i, t_idx, alpha))

    n_refs_avg = float(np.mean([len(r) for r in sample_all_refs if r])) if any(sample_all_refs) else 0.0
    print(f"[SIOP] Pseudo-inputs: {len(all_prompt_ids)} pairs "
          f"(num_refs={num_refs}, avg refs/sample={n_refs_avg:.2f}), "
          f"{len(single_turn_rewards)} single-turn.", flush=True)

    return SiopClusterResult(
        bsz=bsz,
        sample_cluster_freq=sample_cluster_freq,
        sample_reference=sample_reference,
        pseudo_prompt_ids=all_prompt_ids,
        pseudo_response_ids=all_response_ids,
        pseudo_ref_ranges=all_ref_ranges,
        sample_turn_map=sample_turn_map,
        single_turn_rewards=single_turn_rewards,
        siop_lambda=siop_lambda,
        siop_epsilon=siop_epsilon,
    )


# ---------------------------------------------------------------------------
# Phase 2: Score via external vLLM server
# ---------------------------------------------------------------------------

def score_pseudo_inputs(cluster_result: SiopClusterResult) -> list[float]:
    """Send pseudo-inputs to the SIOP scoring server.

    Returns zeros if the server is unavailable (graceful degradation).
    """
    if not cluster_result.pseudo_prompt_ids:
        return []

    from verl.utils.siop.scoring_client import get_siop_scoring_client
    client = get_siop_scoring_client()

    if client is None:
        n = len(cluster_result.pseudo_prompt_ids)
        print(f"[SIOP] Scoring server unavailable — {n} pseudo-inputs scored as 0.", flush=True)
        return [0.0] * n

    items = [
        {
            "prompt_ids": p,
            "response_ids": r,
            "ref_start": rs,
            "ref_end": re_,
        }
        for p, r, (rs, re_) in zip(
            cluster_result.pseudo_prompt_ids,
            cluster_result.pseudo_response_ids,
            cluster_result.pseudo_ref_ranges,
        )
    ]
    return client.score_batch(items)


# ---------------------------------------------------------------------------
# Phase 3: Log-probs → process rewards
# ---------------------------------------------------------------------------

def compute_siop_process_rewards(
    cluster_result: SiopClusterResult,
    mean_log_probs: list[float],
) -> np.ndarray:
    """Compute per-sample process rewards from mean log-probs."""
    bsz = cluster_result.bsz
    lam = cluster_result.siop_lambda
    eps = cluster_result.siop_epsilon

    rewards = np.empty(bsz, dtype=object)
    for i in range(bsz):
        rewards[i] = []

    # Single-turn: terminal reward only
    for idx, r in cluster_result.single_turn_rewards.items():
        rewards[idx] = r

    if not cluster_result.sample_turn_map:
        n = sum(1 for r in rewards if len(r) > 0)
        print(f"[SIOP] No multi-turn inputs. {n}/{bsz} have terminal rewards.", flush=True)
        return rewards

    # Group by (sample, turn). turn_idx=-1 is the prompt-only baseline (τ_0).
    # Each cell holds a list of (alpha, mlp) across the cluster's reference set.
    sample_turns: dict[int, dict[int, list]] = {}
    for entry_idx, (sample_idx, turn_idx, alpha) in enumerate(cluster_result.sample_turn_map):
        sample_turns.setdefault(sample_idx, {}).setdefault(turn_idx, []).append(
            (alpha, mean_log_probs[entry_idx])
        )

    def _aggregate_p_hat(entries):
        """Weighted sum: p̂ = Σ α · exp(mlp). Returns None if no finite entries."""
        total = 0.0
        any_finite = False
        for alpha, mlp in entries:
            if math.isnan(mlp) or math.isinf(mlp):
                continue
            total += alpha * math.exp(mlp)
            any_finite = True
        return total if any_finite else None

    for sample_idx, turn_dict in sample_turns.items():
        sorted_turns = sorted(turn_dict.keys())  # -1 first, then 0,1,2,...
        q_c = cluster_result.sample_cluster_freq[sample_idx]

        # Compute p̂ per turn (including baseline τ_0)
        p_hats = [(tidx, _aggregate_p_hat(turn_dict[tidx])) for tidx in sorted_turns]

        rews = []
        for t in range(len(p_hats)):
            tidx, p_t = p_hats[t]
            if tidx == -1:
                continue
            _, p_prev = p_hats[t - 1]
            if p_t is None or p_prev is None:
                rews.append(0.0)
                continue
            r = q_c * (math.log(p_t + eps) - math.log(p_prev + eps))
            rews.append(0.0 if (math.isnan(r) or math.isinf(r)) else r)

        # Augment final turn: r̃_T += (1 - λ) * q(c|q)
        if rews:
            rews[-1] += (1.0 - lam) * q_c

        rewards[sample_idx] = rews

    n = sum(1 for r in rewards if len(r) > 0)
    # Debug: print reward distribution for first few multi-turn samples
    debug_count = 0
    for sample_idx, turn_dict in sample_turns.items():
        if debug_count >= 3:
            break
        rews = rewards[sample_idx]
        q_c = cluster_result.sample_cluster_freq[sample_idx]
        sorted_turns = sorted(turn_dict.keys())
        p_hat_dbg = [(t, _aggregate_p_hat(turn_dict[t])) for t in sorted_turns]
        print(f"[SIOP-debug] sample={sample_idx} q_c={q_c:.3f} "
              f"n_refs={len(turn_dict[sorted_turns[0]])} "
              f"p_hats={[(t, f'{v:.4f}' if v is not None else 'None') for t, v in p_hat_dbg]} "
              f"rewards={[f'{r:.4f}' for r in rews]}", flush=True)
        debug_count += 1

    print(f"[SIOP] Process rewards computed. {n}/{bsz} samples.", flush=True)
    return rewards


# ---------------------------------------------------------------------------
# TTRL baseline: hard majority-vote binary reward (no scoring server)
# ---------------------------------------------------------------------------

def compute_ttrl_majority_rewards(batch, tokenizer, config) -> np.ndarray:
    """TTRL: per-rollout reward = 1 if the rollout's semantic cluster is the
    dominant cluster for its prompt group, 0 otherwise.  Reuses SIOP's NLI
    clustering; skips the scoring server.  Outcome-only; expected to be paired
    with broadcast advantage (``siop_advantage_mode=broadcast``)."""
    cluster_result = siop_cluster_pass(batch, tokenizer, config)
    bsz = cluster_result.bsz
    index = batch.non_tensor_batch.get("uid", np.arange(bsz, dtype=object))

    groups: dict[str, list[int]] = {}
    for i in range(bsz):
        groups.setdefault(str(index[i]), []).append(i)

    majority_ref: dict[str, str] = {}
    for uid, idxs in groups.items():
        counts: dict[str, int] = {}
        for i in idxs:
            ref = cluster_result.sample_reference[i]
            if ref:
                counts[ref] = counts.get(ref, 0) + 1
        if counts:
            majority_ref[uid] = max(counts, key=counts.get)

    rewards = np.empty(bsz, dtype=object)
    n_win = 0
    for i in range(bsz):
        ref_i = cluster_result.sample_reference[i]
        uid = str(index[i])
        win = bool(ref_i and ref_i == majority_ref.get(uid))
        rewards[i] = [1.0 if win else 0.0]
        if win:
            n_win += 1

    print(f"[TTRL] majority-vote binary: {n_win}/{bsz} rollouts in majority cluster "
          f"({len(groups)} groups)", flush=True)
    return rewards


# ---------------------------------------------------------------------------
# EMPO baseline: soft cluster-frequency reward (no scoring server)
# ---------------------------------------------------------------------------

def compute_empo_cluster_freq_rewards(batch, tokenizer, config) -> np.ndarray:
    """EMPO (Zhang et al. 2025): per-rollout reward = p(c|q) ≈ |c|/K, where
    c is the rollout's semantic cluster.  Outcome-only; pairs with broadcast
    advantage.  Minimizing -E[r] is equivalent to minimizing semantic entropy
    H = -Σ p(c) log p(c) in expectation."""
    cluster_result = siop_cluster_pass(batch, tokenizer, config)
    bsz = cluster_result.bsz

    rewards = np.empty(bsz, dtype=object)
    nonzero = 0
    for i in range(bsz):
        freq = float(cluster_result.sample_cluster_freq[i])
        rewards[i] = [freq]
        if freq > 0:
            nonzero += 1

    mean_freq = float(np.mean([float(f) for f in cluster_result.sample_cluster_freq]))
    print(f"[EMPO] cluster-freq rewards: mean={mean_freq:.3f}, "
          f"nonzero={nonzero}/{bsz}", flush=True)
    return rewards


# ---------------------------------------------------------------------------
# All-in-one entry point (called from ray_trainer)
# ---------------------------------------------------------------------------

def compute_siop_rewards_full(batch, tokenizer, config) -> np.ndarray:
    """Dispatch on ``siop_reward_mode``.

    - ``process`` (default): three-phase SIOP pipeline (cluster → score → diff).
    - ``majority_binary``: TTRL hard-majority baseline (cluster only, binary).
    - ``cluster_frequency``: EMPO soft cluster-frequency baseline.
    """
    reward_mode = getattr(config, "siop_reward_mode", "process")
    if reward_mode == "majority_binary":
        return compute_ttrl_majority_rewards(batch, tokenizer, config)
    if reward_mode == "cluster_frequency":
        return compute_empo_cluster_freq_rewards(batch, tokenizer, config)
    cluster_result = siop_cluster_pass(batch, tokenizer, config)
    mean_log_probs = score_pseudo_inputs(cluster_result)
    return compute_siop_process_rewards(cluster_result, mean_log_probs)
