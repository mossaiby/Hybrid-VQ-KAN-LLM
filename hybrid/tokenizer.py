"""Byte-level BPE tokenizer with individual-digit splitting and the hybrid special tokens."""
from dataclasses import dataclass

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<latent>", "</latent>", "<slot>"]


@dataclass(frozen=True)
class Specials:
    pad: int
    unk: int
    bos: int
    eos: int
    lat_open: int
    lat_close: int
    slot: int  # placeholder id marking a continuous latent input position; never predicted

    def banned_in_text(self):
        """Token ids the text-mode sampler must never emit."""
        return [self.pad, self.unk, self.bos, self.lat_close, self.slot]


def train_tokenizer(text_iter, vocab_size, path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),  # helps arithmetic a lot at small scale
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iter, trainer=trainer)
    tok.save(str(path))
    return tok


def load_tokenizer(path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    ids = [tok.token_to_id(s) for s in SPECIAL_TOKENS]
    if any(i is None for i in ids):
        raise ValueError(f"Tokenizer at {path} is missing special tokens {SPECIAL_TOKENS}")
    return tok, Specials(*ids)


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def decode(tok, ids):
    return tok.decode(list(ids), skip_special_tokens=False)
