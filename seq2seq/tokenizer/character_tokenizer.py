from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}’•–í€óá«»… º◦©ö°äµ—ø­·òãñ―½¼γ®⇒²▪−√¥£¤ß´úª¾є™，ﬁõ  �►□′″¨³‑¯≈ˆ§‰●ﬂ⇑➘①②„≤±†✜✔➪✖◗¢ไทยếệεληνικαåşıруский 한국어汉语ž¹¿šćþ‚‛─÷〈¸⎯×←→∑δ■ʹ‐≥τ;∆℡ƒð¬¡¦βϕ▼⁄ρσ⋅≡∂≠π⎛⎜⎞ω∗"""
        self.vocab  = {c:i for i, c in enumerate(self.characters)}
        if verbose:
            print("Vocabulary:", self.vocab)

        #raise NotImplementedError("Need to implement vocab initialization")

    def encode(self, text: str) -> torch.Tensor:
        etext = [self.vocab[t] for t in text.lower()]
        return torch.tensor(etext)

    def decode(self, tokens: torch.Tensor) -> str:
        self.newvocab = {i:c for i,c in enumerate(self.characters)}
        dtext = [self.newvocab[int(i)] for i in tokens]
        return "".join(dtext)
