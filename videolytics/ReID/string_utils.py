"""Collection of various operation with words"""


import typing
import re
from . import utils


class CaseTransformer:
    """Class for transforming a word from one case type such as camelCase to other, such as PascalCase"""
    def __init__(self, words: typing.Iterable[str]):
        self.words: typing.List[str] = list(words)

    @staticmethod
    def from_pascal(input_string: str) -> 'CaseTransformer':
        if input_string[0].upper() != input_string[0]:
            raise ValueError(
                "Word {} is not is pascal case (does not start with upper-cased letter)".format(repr(input_string)))
        capital_indices = (match.start() for match in re.finditer('[A-Z]', input_string))
        return CaseTransformer(
            input_string[start].lower() + input_string[start+1:stop]
            for start, stop in utils.shift_zip(capital_indices, sufix=None)
        )

    @staticmethod
    def from_camel(input_string: str) -> 'CaseTransformer':
        if input_string[0].upper() == input_string[0]:
            raise ValueError(
                "Word {} is not is camel case (it starts with upper-cased letter)".format(repr(input_string)))
        capital_indices = (match.start() for match in re.finditer('[A-Z]', input_string))
        return CaseTransformer(
            input_string[start].lower() + input_string[start+1:stop] for start, stop
            in utils.shift_zip(capital_indices, prefix=0, sufix=None)
        )

    @staticmethod
    def from_kebab(input_string: str) -> 'CaseTransformer':
        return CaseTransformer(input_string.split('-'))

    @staticmethod
    def from_snake(input_string: str) -> 'CaseTransformer':
        return CaseTransformer(input_string.split('_'))

    def to_pascal(self) -> str:
        def iterator():
            for word in self.words:
                yield word[0].upper()
                yield word[1:]
        return ''.join(iterator())

    def to_camel(self) -> str:
        def iterator():
            word_iterator = iter(self.words)
            yield next(word_iterator)
            for word in word_iterator:
                yield word[0].upper()
                yield word[1:]
        return ''.join(iterator())

    def to_kebab(self):
        return '-'.join(self.words)

    def to_snake(self):
        return '_'.join(self.words)


def pluralize(string: str):
    if string[-1] == 'y':
        return string[:-1] + 'ies'
    if string[-1] == 's':
        return string + 'es'
    return string + 's'
