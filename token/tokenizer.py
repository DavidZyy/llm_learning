
def merges(ids: list[int], pair: tuple, idx: int) -> list[int]:
    """
    input:
        ids: a list of int
        pair: a tuple of int
        idx: an int
    merge a pair in the sequence with a new index, and return the merged sequence
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def convert_text_to_ids(file_path: str) -> list[int]:
    """
    input:
        file_path: a string
    output:
        a list of int ids
    use utf-8 to encode text
    """
    # read from file_path, get a string:
    text = open(file_path, 'r', encoding='utf-8').read()
    ids = text.encode("utf-8")
    ids = list(map(int, ids))
    return ids


def get_stats(ids: list[int]) -> dict[tuple, int]:
    """
    input:
        ids: a list of int
    return:
        a dictionary of pairs and its appear times
    count the times of pairs appeared in a sentence
    """
    stats = {}
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


class Tokenizer:
    def __init__(self):
        self.merges: dict[tuple, int] = {}  # (int, int) -> int, use train to get merges
        self.vocab: dict[int, bytes] = {}  # (int, int) -> int, use train to get merges
        # self.vocab_size = vocab_size

    def train(self, train_text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        input:
            train_text: a string
            vocab_size: an int
            verbose: a bool
        return:
            None
        train the token with a text file
        """
        num_merges = vocab_size - 256
        ids = convert_text_to_ids(train_text)
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)  # get the pair with max appear times
            idx = i + 256
            ids = merges(ids, pair, idx)  # update the ids
            self.merges[pair] = idx  # update the merges
            if verbose:
                print(f"{ i +1}/{num_merges}: {pair} -> {idx}")

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text: str) -> list[int]:
        """
        input: text: a string
        return: a list of int
        encode a string to a list of int
        """
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            # the lambda function: for each pair in stats, return its mapping value in merges, if not found, return inf
            # and then sort the pair in this order
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merges(tokens, pair, idx)
        return tokens

    def decode(self, ids: list[int]) -> str:
        """
        input:
            tokens: a list of int
        return: a string
        decode a list of int to a string
        """
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text


if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.train("./input.txt", vocab_size=276, verbose=True)

    print(tokenizer.decode(tokenizer.encode("hello world")))
