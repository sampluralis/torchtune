# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from typing import Union

import PIL

import torch

from lm_eval.evaluator import evaluate
from lm_eval.models.hf_vlms import HFMultimodalLM
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.utils import make_table
from omegaconf import DictConfig
from torchtune import config, training, utils
from torchtune.data import (
    format_content_with_images,
    left_pad_sequence,
    Message,
    padded_collate_tiled_images_and_mask,
)
from torchtune.generation import generate, sample
from torchtune.modules import TransformerDecoder
from torchtune.modules.common_utils import local_kv_cache
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.modules.transforms import Transform

from torchtune.modules.transforms.tokenizers import (
    HuggingFaceModelTokenizer,
    ModelTokenizer,
)
from torchtune.recipe_interfaces import EvalRecipeInterface
from torchtune.training import FullModelTorchTuneCheckpointer

import csv
import json
from typing import Any, Dict, List, Tuple, Optional

def _safe_get_question(sample: Dict[str, Any]) -> str:
    doc = sample.get("doc", {})
    # Common fields
    for key in ["question", "context", "prompt", "input", "text"]:
        if isinstance(doc, dict) and key in doc and doc[key]:
            return str(doc[key])
    # Fallbacks: try first request's context
    args = sample.get("arguments") or []
    if args and isinstance(args, list) and len(args[0]) >= 1:
        ctx = args[0][0]
        return str(ctx) if ctx is not None else ""
    return ""

def _get_mc_choices_and_gold(doc: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """Return (choices, gold_indices). Supports mc2 first (multi-label), then mc1."""
    choices, gold = [], []
    if isinstance(doc, dict):
        if "mc2_targets" in doc:
            t = doc["mc2_targets"] or {}
            choices = list(t.get("choices", []) or [])
            labels = list(t.get("labels", []) or [])
            gold = [i for i, v in enumerate(labels) if v]
            return choices, gold
        if "mc1_targets" in doc:
            t = doc["mc1_targets"] or {}
            choices = list(t.get("choices", []) or [])
            labels = list(t.get("labels", []) or [])
            gold = [i for i, v in enumerate(labels) if v]
            return choices, gold
    return choices, gold

def _infer_choices_from_arguments(arguments: List[Any]) -> List[str]:
    """When doc lacks mc1/mc2, fall back to the continuation strings in arguments."""
    conts = []
    for a in arguments or []:
        # Instance.args usually like (context, continuation)
        if isinstance(a, (list, tuple)) and len(a) >= 2:
            conts.append(str(a[1]))
    return conts

def _pick_pred_idx_from_scores(filtered_resps: List[Any]) -> Optional[int]:
    """
    filtered_resps is typically a list of tuples like [(score, flag), ...].
    Return argmax index over score.
    """
    try:
        if not filtered_resps:
            return None
        scores = []
        for r in filtered_resps:
            # r could be (score, bool) or [[(score, bool)]] depending on nesting
            if isinstance(r, (list, tuple)) and len(r) >= 1:
                # Flatten one level if needed (e.g., [(-5.1, False)] )
                x = r[0] if (len(r) == 1 and isinstance(r[0], (list, tuple))) else r
                score = float(x[0])
                scores.append(score)
            else:
                return None
        return max(range(len(scores)), key=lambda i: scores[i])
    except Exception:
        return None

def save_eval_qas(output: Dict[str, Any], filepath: str = "eval_qas.csv") -> None:
    """
    Writes a CSV with columns:
      task, question, choices, model_pick_idx, model_pick_text, gold_indices, gold_texts, raw_filtered_resps
    Works for multiple-choice tasks and degrades gracefully for generation tasks.
    """
    samples_by_task = output.get("samples", {})
    rows = []

    for task_name, task_samples in samples_by_task.items():
        for s in task_samples:
            doc = s.get("doc", {}) or {}
            question = _safe_get_question(s)

            # Choices & gold (from doc if present; else from arguments)
            choices, gold_idx = _get_mc_choices_and_gold(doc)
            if not choices:
                choices = _infer_choices_from_arguments(s.get("arguments", []))

            # Predicted index from filtered_resps (log-likelihood scores)
            filtered = s.get("filtered_resps", [])
            pred_idx = _pick_pred_idx_from_scores(filtered)

            # Predicted text
            pred_text = None
            if pred_idx is not None and 0 <= pred_idx < len(choices):
                pred_text = choices[pred_idx]
            else:
                # Generation-style fallback: try to surface produced text
                # Sometimes in lm-eval, 'resps' may already be decoded strings for gen tasks
                resps = s.get("resps", [])
                if resps and isinstance(resps[0], str):
                    pred_text = resps[0]
                elif isinstance(resps, list) and resps and isinstance(resps[0], list) and resps[0] and isinstance(resps[0][0], str):
                    pred_text = resps[0][0]

            # Gold texts
            gold_texts = [choices[i] for i in gold_idx if 0 <= i < len(choices)] if choices and gold_idx else []

            rows.append({
                "task": task_name,
                "question": question,
                "choices": json.dumps(choices, ensure_ascii=False),
                "model_pick_idx": pred_idx if pred_idx is not None else "",
                "model_pick_text": pred_text if pred_text is not None else "",
                "gold_indices": json.dumps(gold_idx),
                "gold_texts": json.dumps(gold_texts, ensure_ascii=False),
                "raw_filtered_resps": json.dumps(filtered),
            })

    fieldnames = ["task", "question", "choices", "model_pick_idx", "model_pick_text", "gold_indices", "gold_texts", "raw_filtered_resps"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[save_eval_qas] Wrote {len(rows)} rows to {filepath}")


class _VLMEvalWrapper(HFMultimodalLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/meta-pytorch/gpt-fast/blob/main/eval.py.

    Note:
        This is ONLY for vision-language models.

    Args:
        model (DeepFusionModel): The VLM to evaluate.
        transform (Transform): The transform (tokenizer) to use for preprocessing.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length.
        batch_size (int): The batch size.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
        image_tag (str): The string to use for the image token. Default is "<image>", which
            is the default used by the MMMU dataset.
        max_images_per_sample (int): The maximum number of images per sample. Defaults to
            the max number of images in MMMU.
    """

    def __init__(
        self,
        model: DeepFusionModel,
        transform: Transform,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        enable_kv_cache: bool = True,
        # TODO (@joecummings): Update these defaults once more multimodal
        # tasks are added to the eval harness
        image_tag: str = "<image>",
        max_images_per_sample: int = 7,
    ):
        self._model = model
        self._transform = transform
        self._device = device
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        # Defaulting KV cache to True for multimodal
        self._enable_kv_cache = True
        self._image_tag = image_tag
        self._max_images_per_sample = max_images_per_sample

    @property
    def model(self):
        # Not actually changing the dtype here, just adding it as a
        # property on the model
        self._model.dtype = self._dtype
        return self._model

    @property
    def model_transform(self):
        return self._transform

    @property
    def device(self):
        return self._device

    @property
    def cache_hook(self):
        # Dummy class to appease the Harness
        class DummyCacheHook:
            def __init__(self):
                self.add_partial = lambda x, y, z: True

        return DummyCacheHook()

    @property
    def rank(self):
        # Hardcoded for now b/c we only support single GPU eval
        return 0

    @property
    def world_size(self):
        # Hardcoded for now b/c we only support single GPU eval
        return 1

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def eos_token_id(self):
        return self._transform.tokenizer.eos_id

    @property
    def eot_token_id(self):
        return self._transform.tokenizer.eot_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def truncation(self):
        return True

    def tok_encode(self, string, **kwargs) -> list[int]:
        # This is only used to get a number of tokens for use in sorting samples in dataset
        # These values will not actually be used for eval
        return self._transform.tokenizer.encode(string, add_bos=False, add_eos=False)

    def tok_decode(self, tokens, skip_special_tokens=True) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._transform.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens
        )

    def tok_batch_multimodal_encode(
        self,
        all_texts: list[str],
        all_images: list[list[PIL.Image.Image]],
        left_truncate_len: int = None,
        *args,
        **kwargs,
    ):
        # Eleuther already parses out the text and images, so we just need to get
        # it into a Message format for our tokenizer
        all_encoded_messages = []

        for text, images in zip(all_texts, all_images):
            # Ensure images are all RGB
            proper_images = []
            for image in images:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                proper_images.append(image)

            # Construct the messages
            messages = []
            content = format_content_with_images(
                text, image_tag=self._image_tag, images=proper_images
            )
            messages.append(Message(role="user", content=content))
            messages.append(Message(role="assistant", content=""))

            # Transform the messages
            tok_batch = self.model_transform({"messages": messages}, inference=True)
            all_encoded_messages.append(tok_batch)

        # Pad the encoded messages
        tok_batch = padded_collate_tiled_images_and_mask(
            all_encoded_messages,
            pad_direction="left",
            pad_max_images=self._max_images_per_sample,
            pad_max_tiles=self._transform.max_num_tiles,
        )
        utils.batch_to_device(tok_batch, self.device)

        # Convert the batch to the format expected by the HF
        tok_batch["input_ids"] = tok_batch.pop("tokens")

        # the harness will use left_truncate_len to indicate that the current batch
        # needs to be truncated to self.max_seq_len - self.max_gen_toks
        if left_truncate_len is not None:
            tok_batch["input_ids"] = tok_batch["input_ids"][:, -left_truncate_len:]

        return tok_batch

    @torch.inference_mode()
    def _model_multimodal_generate(
        self,
        batch: dict[str, torch.Tensor],
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ):
        # 1. Validate inputs
        prompt = batch.pop("input_ids")
        bsz, seq_len = prompt.shape

        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )

        if bsz > 1:
            raise ValueError(
                f"Got a batch size of '{bsz}'. Batch size > 1 is not yet supported for "
                "multimodal generation."
            )

        encoder_max_seq_len = (
            self.model_transform.image_seq_len * self._max_images_per_sample
        )
        # Setup masks for bsz 1
        with self.device:
            causal_mask = torch.tril(
                torch.ones(
                    size=(self.max_length, self.max_length),
                    dtype=torch.bool,
                )
            )
            input_pos = torch.arange(self.max_length)

        batch["input_pos"] = input_pos[None, :seq_len]
        batch["mask"] = causal_mask[None, :seq_len]

        # 2. Setup KV cache
        with local_kv_cache(
            self.model,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self._dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=self.max_length,
        ):
            # 3. Prefill step
            generated_tokens = []
            logits = self.model(prompt, **batch)[:, -1]
            token = sample(logits, temperature=0.0, top_k=None)
            generated_tokens.append(token.item())

            cache_mask = batch["encoder_mask"][:, -1:]

            # 4. Continue generating
            for _ in range(max_length):
                if token.item() in self.model_transform.stop_tokens:
                    break
                logits = self.model(
                    token,
                    mask=causal_mask[None, seq_len, None, :],
                    encoder_input=None,
                    encoder_mask=cache_mask,
                    input_pos=input_pos[None, seq_len],
                )[:, -1]
                token = sample(logits, temperature=0.0, top_k=None)
                generated_tokens.append(token.item())
                seq_len += 1

        # 5. Return generated tokens
        return torch.tensor(generated_tokens, dtype=torch.int32).unsqueeze(0)


def _extract_ctx_cont(req):
    """
    Works with:
      - lm-eval Instance objects (req.args -> (context, continuation))
      - legacy (context, continuation) tuples
      - dict-like fallbacks
    """
    ctx = cont = None
    # Newer harness: Instance with .args
    if hasattr(req, "args"):
        args = getattr(req, "args", ())
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            ctx, cont = args[0], args[1]
    # Legacy: tuple
    elif isinstance(req, (list, tuple)) and len(req) >= 2:
        ctx, cont = req[0], req[1]
    # Fallback: dict-like
    elif isinstance(req, dict):
        ctx = req.get("context", None)
        cont = req.get("continuation", None)
    return ctx, cont

def _render_chat_with_tokenizer(tokenizer, messages, add_generation_prompt=True):
    """
    Uses HF chat_template if available (Llama 3 instruct tokenizers have it).
    Returns a single rendered string with roles + special tokens.
    """
    # If you're wrapping Torchtune's HuggingFaceModelTokenizer:
    base = getattr(tokenizer, "base_tokenizer", None) or tokenizer
    if hasattr(base, "apply_chat_template"):
        return base.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    # Fallback: if your custom tokenizer exposes a prompt_template callable
    if hasattr(tokenizer, "prompt_template"):
        return tokenizer.prompt_template(messages)
    # Else: caller should fall back to manual renderer (see Option B)
    raise RuntimeError("No chat_template available on this tokenizer.")

class _LLMEvalWrapper(HFLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/meta-pytorch/gpt-fast/blob/main/eval.py.

    Note:
        This is for text-only decoder models.

    Args:
        model (TransformerDecoder): The model to evaluate.
        tokenizer (ModelTokenizer): Tokenizer associated with the model being evaluated.
            This should be the same tokenizer used when fine-tuning the model.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length to use.
        batch_size (int): The batch size per GPU to use.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: ModelTokenizer,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.float32,
        enable_kv_cache: bool = True,
    ):
        # TODO (@joecummings): Remove this init function so we don't load in extraneous stuff
        super().__init__(pretrained="gpt2", device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        self._enable_kv_cache = enable_kv_cache
        # Set device explicitely here since HPU is not included in
        # `device_list` in `HFLM` class
        self._device = torch.device(device)

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def enable_kv_cache(self):
        return self._enable_kv_cache

    def loglikelihood(self, requests):
        

        for i, req in enumerate(requests):
            ctx, cont = _extract_ctx_cont(req)
            # ctx/cont are plain strings (provided by harness), so print directly
            print("\n=== SAMPLE", i, "===")
            print("Q (context):", ctx if ctx is not None else "<missing>")
            print("A (continuation):", cont if cont is not None else "<missing>")
        return super().loglikelihood(requests)
    def tok_encode(self, text: str, **kwargs) -> list[int]:
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/meta-pytorch/gpt-fast/blob/main/eval.py#L123.
       # print(f'begin {text} end')
        if isinstance(self._tokenizer, HuggingFaceModelTokenizer):
            return self._tokenizer.base_tokenizer.encode(
                text=text, add_bos=False, add_eos=False
            )
        return self._tokenizer.encode(text=text, add_bos=False, add_eos=False)

    def tok_batch_encode(
        self, text: list[str], left_truncate_len: int = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # for i, t in enumerate(text):
        #     print(f"\n=== SAMPLE {i} PROMPT ===\n{t}\n")
        tokenized_text = [self.tok_encode(x) for x in text]

        # pad left
        x = left_pad_sequence(
            [torch.tensor(x) for x in tokenized_text],
            batch_first=True,
            padding_value=self._tokenizer.pad_id,
        )

        # the harness will use left_truncate_len to indicate that the current batch
        # needs to be truncated to self.max_seq_len - self.max_gen_toks
        if left_truncate_len is not None:
            x = x[:, -left_truncate_len:]

        return x, torch.ones_like(x)  # return 'mask' b/c it's expected by the harness

    def tok_decode(self, tokens: Union[list[int], int], **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model(inps)

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        if hasattr(self._tokenizer, "prompt_template"):
            return self._tokenizer.prompt_template(chat_history)
        if isinstance(self.tokenizer, HuggingFaceModelTokenizer):
            return self.tokenizer.render_template(chat_history)
        raise ValueError(
            "You can't use a tokenizer without a prompt template and apply_chat_template: True. "
            "Use HuggingFaceModelTokenizer if you do not require a custom one."
        )

    @torch.inference_mode()
    def _model_generate(
        self, context: torch.Tensor, **generation_kwargs
    ) -> torch.Tensor:
        bsz, seq_len = context.shape
        print("$##############################")
        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )

        # if we've recieved fewer than self._batch_size samples in the current
        # batch we need to pad the batch out. here we're padding the end of the
        # current batch to the correct length. this is because when we use static
        # KV-caches, the model will expect a fixed batch size for all samples.
        maybe_padded_context = torch.nn.functional.pad(
            context,
            (0, 0, 0, self._batch_size - bsz),
            value=self._tokenizer.eos_id,  # pad with one of the tokenizer's stop tokens so generation can stop early
        )
        with local_kv_cache(
            self.model,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self._dtype,
            decoder_max_seq_len=self.max_length,
        ):
            toks, _ = generate(
                self.model,
                maybe_padded_context,
                max_generated_tokens=self.max_gen_toks,
                temperature=temperature,
                top_k=None,
                pad_id=self._tokenizer.pad_id,
                stop_tokens=self._tokenizer.stop_tokens,
            )

            print(self.tok_decode(toks[0].tolist()))
        return toks[:bsz]


class EleutherEvalRecipe(EvalRecipeInterface):
    """
    This recipe runs evaluation on a trained model using EleutherAI's eval harness.
    This assumes the user has the EleutherAI eval harness installed. See
    https://github.com/EleutherAI/lm-evaluation-harness for more details.

    Features:
        - Single GPU evaluation. Multi-GPU evaluation is currently not supported.
        - Quantization (for text-only models) is supported.
        - Any task from the EleutherAI eval harness

    We recommend launching evaluation using the tune CLI::

        tune run eleuther_eval --config eleuther_evaluation \
            tasks=["truthfulqa_mc2","hellaswag"] \
            limit=50 \
    """

    def __init__(self, cfg: DictConfig) -> None:
        # Double check we have the right Eval Harness version
        from importlib.metadata import version

        if version("lm-eval") < "0.4.5" or version("lm-eval") > "0.4.8":
            raise RuntimeError(
                "This recipe requires EleutherAI Eval Harness between v0.4.5 - 0.4.8."
                "Please install with `pip install lm-eval==0.4.8`"
            )

        # General variable initialization
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(dtype=cfg.dtype, device=self.device)
        self.logger = utils.get_logger(cfg.get("log_level", "info"))
        training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )

        # Eval specific variables
        self.limit = cfg.limit
        self.tasks = list(cfg.tasks)
        self.batch_size = cfg.batch_size
        self.enable_kv_cache = cfg.get("enable_kv_cache", True)
        self.include_path = cfg.get("include_path", None)
        self.apply_chat_template = cfg.get("chat_template", False)

    def setup(self, cfg: DictConfig) -> None:
        # Initialize quantizer and quantization mode
        quantizer = config.instantiate(cfg.quantizer)
        quantization_mode = training.get_quantizer_mode(quantizer)

        # Load checkpoint
        checkpointer = config.instantiate(cfg.checkpointer)

        # Initialize model
        with training.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg.model)

        # Quantize model if requested
        if quantization_mode is not None:
            if not isinstance(checkpointer, FullModelTorchTuneCheckpointer):
                raise ValueError(
                    "Quantization is only supported for models quantized and saved with the "
                    "FullModelTorchTuneCheckpointer - please ensure you have quantized your "
                    "model and are using the quantized weights!"
                )
            if "qat" in quantization_mode:
                raise ValueError(
                    "You have specified a quantizer with 'QAT' - "
                    "QAT quantizers should only be used during quantization aware training "
                    "and when quantizing models. Please use the corresponding post-training "
                    "quantizer e.g. Int8DynActInt4WeightQuantizer for Int8DynActInt4WeightQATQuantizer."
                )
            model = quantizer.quantize(model)
            model = model.to(device=self.device, dtype=self.dtype)
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)[
                training.MODEL_KEY
            ]
            for k, v in ckpt_dict.items():
                ckpt_dict[k] = v.to(self.device)
            model.load_state_dict(ckpt_dict, assign=True)
        else:
            ckpt_dict = checkpointer.load_checkpoint()[training.MODEL_KEY]
            model.load_state_dict(ckpt_dict)

        # Load model weights into initialized model
        self.logger.info(f"Model is initialized with precision {self.dtype}.")

        # Put model in eval mode.
        # Note: This will not disable the dropout applied in SDPA,
        # see https://github.com/pytorch/pytorch/issues/124464
        model.eval()

        # Initialize tokenizer/transform
        model_transform = config.instantiate(cfg.tokenizer)

        # Finally, we setup the actual EvalWrapper class
        if isinstance(model, DeepFusionModel):
            eleuther_model_wrapper = _VLMEvalWrapper
            if not self.enable_kv_cache:
                self.logger.debug(
                    "Received enable_kv_cache=False, but KV cache is required for running "
                    "multimodal generation in a timely manner. Setting enable_kv_cache=True."
                )
        elif isinstance(model, TransformerDecoder):
            eleuther_model_wrapper = _LLMEvalWrapper
        self.eleuther_model_wrapper = eleuther_model_wrapper(
            model,
            model_transform,
            device=self.device,
            max_seq_length=cfg.max_seq_length,
            batch_size=self.batch_size,
            dtype=self.dtype,
            enable_kv_cache=self.enable_kv_cache,
        )

    def evaluate(self) -> None:
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # Initialize tasks for the harness
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)

        # Run evaluation
        t0 = time.time()
        self.logger.info(f"Running evaluation on the following tasks: {self.tasks}")
        output = evaluate(
            self.eleuther_model_wrapper,
            task_dict,
            apply_chat_template=self.apply_chat_template,
            limit=self.limit,
            log_samples=True,
        )


        t1 = time.time() - t0
        import json

        samples = output.get("samples", {})
        for task_name, task_samples in samples.items():
            print(f"\n=== Task: {task_name} ===")
            for s in task_samples:
                ctx = s.get("doc") or s.get("arguments") or ""
                target = s.get("target")
                resps = s.get("resps")          # raw model scores or outputs
                filtered = s.get("filtered_resps")
                metrics = s.get("metrics")

                print("\nQ:", json.dumps(ctx, ensure_ascii=False))
                print("Target (gold):", target)
                print("Model raw responses:", resps)
                print("Filtered responses:", filtered)
                print("Metrics:", metrics)
        
        save_eval_qas(output, filepath="eval_qas.csv")
        # Log metrics
        self.logger.info(f"Eval completed in {t1:.02f} seconds.")
        if self.device.type != "cpu" and self.device.type != "mps":
            torch_device = utils.get_torch_device_namespace()
            self.logger.info(
                f"Max memory allocated: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )
        formatted_output = make_table(output)
        self.logger.info(f"\n\n{formatted_output}\n")


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    config.log_config(recipe_name="EleutherEvalRecipe", cfg=cfg)
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()


if __name__ == "__main__":
    sys.exit(recipe_main())
