# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth, extra_info=extra_info)

    elif data_source in ["2wiki", "popqa", "tq", "hotpotqa", "Bamboogle", "nq", "musique"]:
        # Check if SIOP process rewards are available
        siop_rewards = None
        if extra_info and isinstance(extra_info, dict):
            siop_rewards = extra_info.get("siop_process_rewards", None)

        if siop_rewards is not None and len(siop_rewards) > 0:
            from .siop_reward import compute_score as siop_compute_score

            # Normalize ground_truth format
            if isinstance(ground_truth, str):
                ground_truth = {"target": ground_truth.split("<|answer_split|>")}
            elif isinstance(ground_truth, dict):
                if "target" in ground_truth:
                    pass
                elif "ground_truth" in ground_truth:
                    gt = ground_truth["ground_truth"]
                    if isinstance(gt, str):
                        ground_truth = {"target": gt.split("<|answer_split|>")}
                    elif isinstance(gt, dict) and "target" in gt:
                        ground_truth = gt
                    else:
                        ground_truth = {"target": [str(gt)]}

            res = siop_compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                data_source=data_source,
                siop_process_rewards=siop_rewards,
                tokenizer=kwargs.get("tokenizer", None),
                is_validation=kwargs.get("is_validation", False),
                response_token_count=extra_info.get("response_token_count", None) if extra_info else None,
                valid_response_mask=extra_info.get("valid_response_mask", None) if extra_info else None,
            )
        else:
            # IGPO datasets: ground_truth may be:
            # - string with <|answer_split|> separator (original format)
            # - {"ground_truth": "ans1<|answer_split|>ans2"} (original dict format)
            # - {"target": ["ans1", "ans2"]} (preprocessed format)
            # - {"ground_truth": {"target": ["ans1", "ans2"]}} (preprocessed nested format)
            from . import search_r1_like_qa_em

            if isinstance(ground_truth, str):
                ground_truth = {"target": ground_truth.split("<|answer_split|>")}
            elif isinstance(ground_truth, dict):
                if "target" in ground_truth:
                    pass  # already in correct format
                elif "ground_truth" in ground_truth:
                    gt = ground_truth["ground_truth"]
                    if isinstance(gt, str):
                        ground_truth = {"target": gt.split("<|answer_split|>")}
                    elif isinstance(gt, dict) and "target" in gt:
                        ground_truth = gt
                    else:
                        ground_truth = {"target": [str(gt)]}
            em_res = search_r1_like_qa_em.compute_score(solution_str, ground_truth, extra_info=extra_info)
            from .siop_reward import compute_f1, extract_answer
            pred = extract_answer(solution_str)
            gt_list = ground_truth.get("target", []) if isinstance(ground_truth, dict) else []
            if isinstance(gt_list, str):
                gt_list = [gt_list]
            f1_val = max((compute_f1(pred, gt) for gt in gt_list), default=0.0) if pred else 0.0
            res = {"score": float(em_res), "acc": float(em_res), "f1": f1_val}
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
