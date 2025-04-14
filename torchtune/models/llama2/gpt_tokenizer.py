from typing import Any, List, Mapping, Optional, Tuple

from transformers import AutoTokenizer

from torchtune.data import Message, PromptTemplate
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import (
    ModelTokenizer,
    tokenize_messages_no_special_tokens,
)


class OPTTokenizer(ModelTokenizer, Transform):
    """
    A Hugging Face-based tokenizer for the OPT (GPT2-style) family of models,
    built in a style similar to the Llama2Tokenizer from Torchtune.

    Args:
        path (str):
            A Hugging Face model name or local path to a tokenizer, e.g. 'facebook/opt-2.7b'.
        max_seq_len (Optional[int]):
            If set, tokens are truncated to this length (using the logic in
            'tokenize_messages_no_special_tokens'). Default: None
        prompt_template (Optional[PromptTemplate]):
            A PromptTemplate to apply to the messages before tokenization.
        truncation_type (str):
            Either 'left' or 'right'. Default is 'right'.

    Usage:
        >>> tokenizer = OPTTokenizer("facebook/opt-2.7b", max_seq_len=2048)
        >>> messages = [
        ...     Message(role="user", content="Hello, how are you?"),
        ...     Message(role="assistant", content="I'm fine, thank you."),
        ... ]
        >>> token_ids, mask = tokenizer.tokenize_messages(messages)
        >>> print(token_ids)
        >>> print(mask)
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
        truncation_type: str = "right",
    ):
        # Load a GPT2-like tokenizer directly from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

        # Some GPT2-based tokenizers lack pad_token, so we set it if needed.
        if self.tokenizer.pad_token is None:
            # For GPT2/OPT, pad_token is often not defined. We can reuse the eos token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.truncation_type = truncation_type

        # We'll define "stop_tokens" as [eos_id], which GPT2/OPT usually uses to stop generation.
        # If you want multiple stopping IDs, append them here.
        self.stop_tokens = [self.eos_id]

    @property
    def eos_id(self) -> int:
        # GPT2-based models typically have an eos_token_id
        return self.tokenizer.eos_token_id

    @property
    def bos_id(self) -> int:
        # GPT2-based models often do not define a "bos_token_id" distinctly.
        # Some do, some do not. If not present, we can just reuse eos or return None.
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.eos_id

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
    ) -> List[int]:
        """
        Encode text into token IDs using the Hugging Face GPT2/OPT tokenizer.
        Optionally add BOS/EOS tokens.

        Args:
            text: The input text to tokenize.
            add_bos: If True, insert bos_id at the start. (GPT2 often ignores this, but included here for parity.)
            add_eos: If True, insert eos_id at the end.
            trim_leading_whitespace: If True, we strip() text from the left before tokenizing.

        Returns:
            A list of token IDs.
        """
        if trim_leading_whitespace:
            text = text.lstrip()

        # Basic encoding without HF "special tokens" (since we manage BOS/EOS ourselves).
        # set `add_special_tokens=False` so we don't get GPT2's default <|endoftext|> automatically.
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        if add_bos and self.bos_id is not None:
            encoded = [self.bos_id] + encoded
        if add_eos and self.eos_id is not None:
            encoded = encoded + [self.eos_id]

        return encoded

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs into text with the HF tokenizer.
        """
        # We do *not* ask to skip special tokens here, so if the user wants to
        # remove <|endoftext|>, we can strip it manually or rely on the defaults.
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize multiple messages by optionally applying a PromptTemplate,
        then encoding each message individually (to preserve spacing
        behavior), and concatenating them. Also returns a "mask" array
        indicating which tokens are masked (i.e., user/system tokens)
        vs. model/assistant tokens, as determined by each Message's "masked" field.

        Args:
            messages: A list of Message objects.
            add_start_tokens: If True, prepend BOS to the first chunk.
            add_end_tokens: If True, append EOS to the last chunk.

        Returns:
            (token_ids, mask):
              - token_ids: List of all tokens concatenated.
              - mask: List of booleans with the same length, indicating
                which tokens are masked (True) vs not (False).
        """
        # If a prompt template is given, apply it here
        templated_messages = (
            self.prompt_template(messages) if self.prompt_template else messages
        )

        # Reuse the same utility as in Llama2Tokenizer:
        # `tokenize_messages_no_special_tokens` handles iterating over messages,
        # applying `encode`, and handling truncation.
        return tokenize_messages_no_special_tokens(
            tokenizer=self,
            messages=templated_messages,
            bos_id=self.bos_id if add_start_tokens else None,
            eos_id=self.eos_id if add_end_tokens else None,
            truncation_type=self.truncation_type,
        )

    def __call__(
        self,
        sample: Mapping[str, Any],
        inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Torchtune's Transform interface: processes a sample with a "messages" field
        by converting it to a list of token IDs ("tokens") and a boolean mask ("mask").

        If inference=True, we typically do NOT add an EOS at the end (so generation
        can continue). If inference=False, we do add EOS.
        """
        messages = sample.pop("messages")
        # By default: for training, add end tokens; for inference, do not.
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
