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

import re
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

_TRACE_BLOCK_PATTERN = re.compile(r"(<think>.*?</think>|<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>|<answer>.*?</answer>)", re.DOTALL)
_EMPTY_TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>\s*</tool_response>", re.DOTALL)


def _maybe_unwrap_scalar(value: Any) -> Any:
    if isinstance(value, (str, bytes, dict, list, tuple)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _stringify_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "image":
                    parts.append("[image]")
                elif item_type == "video":
                    parts.append("[video]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _format_prompt_messages(raw_prompt: Any) -> str:
    raw_prompt = _maybe_unwrap_scalar(raw_prompt)
    if raw_prompt is None:
        return "[prompt]\n<EMPTY>"

    lines = []
    if isinstance(raw_prompt, (list, tuple)):
        for idx, message in enumerate(raw_prompt):
            if not isinstance(message, dict):
                lines.append(f"[context.{idx}]")
                lines.append(str(message))
                continue
            role = message.get("role", f"context.{idx}")
            content = _stringify_message_content(message.get("content"))
            lines.append(f"[{role}]")
            lines.append(content if content.strip() else "<EMPTY>")
    else:
        lines.append("[prompt_raw]")
        lines.append(str(raw_prompt))
    return "\n".join(lines)


def _format_response_trace(response_str: str) -> str:
    text = response_str or ""
    lines = []
    last_end = 0

    for match in _TRACE_BLOCK_PATTERN.finditer(text):
        plain_text = text[last_end:match.start()].strip()
        if plain_text:
            lines.append("[assistant.text]")
            lines.append(plain_text)

        block = match.group(0)
        if block.startswith("<think>"):
            label = "[assistant.think]"
            content = block[len("<think>"):-len("</think>")].strip()
        elif block.startswith("<tool_call>"):
            label = "[assistant.tool_call]"
            content = block[len("<tool_call>"):-len("</tool_call>")].strip()
        elif block.startswith("<tool_response>"):
            label = "[tool.response]"
            content = block[len("<tool_response>"):-len("</tool_response>")].strip()
        else:
            label = "[assistant.answer]"
            content = block[len("<answer>"):-len("</answer>")].strip()

        lines.append(label)
        lines.append(content if content else "<EMPTY>")
        last_end = match.end()

    tail = text[last_end:].strip()
    if tail:
        lines.append("[assistant.text]")
        lines.append(tail)

    if not lines:
        lines.append("[assistant.text]")
        lines.append(text.strip() if text.strip() else "<EMPTY>")

    if _EMPTY_TOOL_RESPONSE_PATTERN.search(text):
        lines.append("[trajectory.warning]")
        lines.append("Found empty <tool_response> block.")

    return "\n".join(lines)


def _format_rollout_trajectory(raw_prompt: Any, response_str: str) -> str:
    return "\n".join([
        _format_prompt_messages(raw_prompt),
        "[rollout.response]",
        _format_response_trace(response_str),
    ])


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_trace_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["prompt_str"] = prompt_str

            # Pass SIOP process rewards if available
            siop_rewards = data_item.non_tensor_batch.get("siop_process_rewards", None)
            if siop_rewards is not None:
                extra_info["siop_process_rewards"] = siop_rewards
                extra_info["response_token_count"] = int(valid_response_length)
                if "response_mask" in data_item.batch.keys():
                    extra_info["valid_response_mask"] = (
                        data_item.batch["response_mask"][:valid_response_length].detach().cpu().tolist()
                    )

            # Debug: log first sample's routing info
            if i == 0:
                has_siop = siop_rewards is not None and len(siop_rewards) > 0 if siop_rewards is not None else False
                print(f"[naive-debug] i=0, data_source={data_source!r}, "
                      f"siop_rewards={siop_rewards}, has_siop={has_siop}, "
                      f"valid_response_length={valid_response_length}", flush=True)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                tokenizer=self.tokenizer,
            )

            # Debug: log first sample's score type
            if i == 0:
                score_type = type(score).__name__
                score_preview = score[:5] if isinstance(score, list) else score
                print(f"[naive-debug] i=0, score type={score_type}, preview={score_preview}", flush=True)

            if isinstance(score, list):
                # Turn-level rewards (e.g., from siop_reward.compute_score)
                # score is a dense per-token list; boundary detection downstream
                # must not rely on nonzero rewards because valid turn rewards can be 0.
                for idx, val in enumerate(score):
                    if idx < reward_tensor.shape[1]:
                        reward_tensor[i, idx] = val
                reward = sum(score)
                # Do NOT overwrite last token — turn-level rewards are already placed
            elif isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                reward_tensor[i, valid_response_length - 1] = reward
            else:
                reward = score
                reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                raw_prompt = data_item.non_tensor_batch.get("raw_prompt", None)
                print("[trajectory]")
                print(_format_rollout_trajectory(raw_prompt=raw_prompt, response_str=response_trace_str))
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
