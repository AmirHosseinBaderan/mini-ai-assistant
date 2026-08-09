import re
import unicodedata
from collections import Counter


class IntentTokenizer:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        min_frequency: int = 1,
    ):
        self.min_frequency = min_frequency

        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        self.id_to_token: dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def normalize(self, text: str) -> str:
        """
        Normalize Unicode text.

        NFC keeps characters in a canonical Unicode representation
        without removing Persian-specific characters such as ZWNJ.
        """
        return unicodedata.normalize("NFC", text)

    def tokenize(self, text: str) -> list[str]:
        """
        Convert text into word/punctuation tokens.

        Examples:

            "What is Python?"
            -> ["what", "is", "python", "?"]

            "پایتون چیست؟"
            -> ["پایتون", "چیست", "؟"]
        """

        text = self.normalize(text)
        text = text.lower().strip()

        return re.findall(
            r"\w+|[^\w\s]",
            text,
            flags=re.UNICODE,
        )

    def build_vocab(self, texts: list[str]) -> None:
        """
        Build vocabulary from training texts.
        """

        counter = Counter()

        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        for token, frequency in counter.items():

            if frequency < self.min_frequency:
                continue

            if token in self.token_to_id:
                continue

            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def encode(
        self,
        text: str,
        max_length: int,
    ) -> list[int]:
        """
        Convert text into padded token IDs.

        Output length is always exactly max_length.
        """

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero"
            )

        tokens = self.tokenize(text)

        token_ids = [
            self.token_to_id.get(
                token,
                self.unk_id,
            )
            for token in tokens
        ]

        # Truncation
        token_ids = token_ids[:max_length]

        # Padding
        padding_length = max_length - len(token_ids)

        token_ids.extend(
            [self.pad_id] * padding_length
        )

        return token_ids

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        """
        Convert token IDs back into text.

        PAD tokens are ignored.
        Unknown IDs are represented as <UNK>.
        """

        tokens = []

        for token_id in token_ids:

            if token_id == self.pad_id:
                continue

            token = self.id_to_token.get(
                token_id,
                self.UNK_TOKEN,
            )

            tokens.append(token)

        return " ".join(tokens)
