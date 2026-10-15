import unittest

from tokenization import UNK_TOKEN, WhitespaceTokenizer


class WhitespaceTokenizerTests(unittest.TestCase):
    def test_splits_punctuation_into_separate_tokens(self) -> None:
        tokenizer = WhitespaceTokenizer().fit(["Hello, world!"])

        self.assertEqual(tokenizer.tokenize("Hello, world!"), ["Hello", ",", "world", "!"])

    def test_maps_unseen_tokens_to_unknown_id(self) -> None:
        tokenizer = WhitespaceTokenizer(vocab_size=2).fit(["known known"])

        self.assertEqual(tokenizer.tokenize("known novel."), ["known", UNK_TOKEN, UNK_TOKEN])
        self.assertEqual(tokenizer.encode("known novel."), [1, 0, 0])

    def test_vocabulary_is_frequency_ranked_and_capped(self) -> None:
        tokenizer = WhitespaceTokenizer(vocab_size=3).fit(["pear pear apple plum"])

        self.assertEqual(tokenizer.vocabulary, (UNK_TOKEN, "pear", "apple"))

    def test_word_oov_statistics_ignore_punctuation(self) -> None:
        tokenizer = WhitespaceTokenizer().fit(["known"])

        self.assertEqual(
            tokenizer.word_oov_statistics("Known, unknown!"),
            {"total_words": 2, "oov_words": 2, "oov_percentage": 100.0},
        )

    def test_rejects_nonpositive_vocabulary_size(self) -> None:
        with self.assertRaises(ValueError):
            WhitespaceTokenizer(vocab_size=0)

    def test_requires_fit_before_use(self) -> None:
        tokenizer = WhitespaceTokenizer()

        with self.assertRaises(RuntimeError):
            tokenizer.tokenize("text")


if __name__ == "__main__":
    unittest.main()
