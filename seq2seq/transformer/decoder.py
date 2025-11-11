import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding
from seq2seq.data.fr_en import tokenizer


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length
        self.dropout = dropout

        # Define any layers you'll need in the forward pass
        self.mha = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.ff = FeedForwardNN(embedding_dim, ffn_hidden_dim)
        self.ln = nn.LayerNorm(self.embedding_dim)


    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """
        m = self.mha(x,x,x,tgt_mask)
        l1 = self.ln(x+m)
        f = self.ff(l1)
        l2 = self.ln(l1+f)

        if enc_x is not None:
            m = self.mha(l2,enc_x,enc_x, src_mask)
            l1 = nn.LayerNorm(self.embedding_dim)
            l1 = l1(x+m)
            f = self.ff(l1)
            l2 = nn.LayerNorm(self.embedding_dim)
            l2 = l2(l1+f)

        return l2


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length
        self.dropout = dropout

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        #
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.layers = nn.ModuleList([DecoderLayer(self.num_heads,self.embedding_dim, self.ffn_hidden_dim,self.qk_length,self.value_length, self.dropout) for _ in range(self.num_layers)])

        self.lastlinear = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x,enc_x, tgt_mask, src_mask)
        x = self.lastlinear(x)
        sm = nn.Softmax(dim=-1)
        x = sm(x)
        return x
