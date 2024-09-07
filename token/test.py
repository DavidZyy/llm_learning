import pytest
import os
# from .tokenizer import Tokenizer  #  only used in packages(a directory have __init__.py)
from tokenizer import Tokenizer

# a few strings to test the tokenizers on
test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]
def unpack(text):
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

@pytest.mark.parametrize("tokenizer_factory", [Tokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    # tokenizer.train(text, 100)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
