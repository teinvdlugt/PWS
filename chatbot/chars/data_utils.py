from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

PAD_SYMBOL = '#'
GO_SYMBOL = '>'
EOS_SYMBOL = '<'
UNK_SYMBOL = '~'

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_VOCAB = ['#', '>', '<', '~']  # Will be the start of TOTAL_VOCAB

VOCAB = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
         'u', 'v', 'w', 'x', 'y', 'z', '.', ',', '!', '?', ' ']
TOTAL_VOCAB = SPECIAL_VOCAB + VOCAB


def sentence_to_token_ids(sentence):
    """Convert a string to list of integers representing character-ids.

    Because this conversational model is character-based, the sentence
    "i have a dog" will become tokenized into ["i", " ", "h", "a", "v", ...]
    and with vocabulary {"i": 1, " ": 0, "h": 2, "a": 4, "v": 7"} this
    function will return [1, 0, 2, 4, 7, ...].

    Args:
      sentence: the sentence to convert to char-ids (string)

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    token_ids = []
    for c in sentence:
        if c in VOCAB:
            token_ids.append(VOCAB.index(c) + len(SPECIAL_VOCAB))
        else:
            token_ids.append(UNK_ID)
    return token_ids
