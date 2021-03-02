"""Collection of various useful functions and classes"""


import argparse
import collections
import datetime
import itertools
import logging
import math
import numbers
import numpy as np
import re
import sqlalchemy.orm
import sys
import typing


class AnnotatedDetection:
    __slots__ = ['camera_id', 'frame_id', 'detection_id', 'class_', 'position', 'dimensions', 'image', 'timestamp', 'feature']

    def __init__(self, camera_id, frame_id, detection_id, class_, position, dimensions, image, timestamp, feature):
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.detection_id = detection_id
        self.class_ = class_
        self.position = position
        self.dimensions = dimensions
        self.image = image
        self.timestamp = timestamp
        self.feature = feature

    def __repr__(self):
        params_str = ['{}={}'.format(param, repr(getattr(self, param))) for param in self.__slots__ if param not in ('image', 'feature')]
        return '{}({})'.format(self.__class__.__name__, params_str)

    @classmethod
    def from_raw(cls, data, camera_id=None):
        dimensions = np.array([data.bottom - data.top, data.right - data.left], dtype=np.uint32)
        position = np.array([data.left, data.top])
        image = np.frombuffer(data.crop, dtype=np.uint8).reshape(*dimensions, 3)
        feature = np.frombuffer(data.feature, dtype=np.float64) if data.feature is not None else None
        return cls(
            camera_id=coalesce(camera_id, getattr(data, 'camera_id', None)),
            frame_id=data.frame_id,
            detection_id=data.detection_id,
            class_=getattr(data, 'class'),
            position=position,
            dimensions=dimensions,
            image=image,
            timestamp=getattr(data, 'timestamp', None),
            feature=feature,
        )

    @property
    def centre_position(self):
        return self.position + self.dimensions / 2

    @property
    def left(self):
        return self.position[0]

    @property
    def right(self):
        return self.position[0] + self.dimensions[1]

    @property
    def top(self):
        return self.position[1]

    @property
    def bottom(self):
        return self.position[1] + self.dimensions[0]


class memoize:
    def __init__(self, func):
        self.func = func
        self.memory = dict()

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise NotImplementedError("Kwargs in memoize decorator is not yet implemented")
        key = args
        if key not in self.memory:
            self.memory[key] = self.func(*args, **kwargs)
        return self.memory[key]


def coalesce(*args):
    for item in args:
        if item is not None:
            return item
    return None


FirstKey = typing.TypeVar("FirstKey")
SecondKey = typing.TypeVar("SecondKey")
AnyKey = typing.Union[FirstKey, SecondKey]
FullKey = typing.Union[typing.Tuple[FirstKey, SecondKey], typing.Tuple[SecondKey, FirstKey]]
Data = typing.TypeVar("Data")


class UnorderedPairDict(collections.UserDict, typing.Generic[FirstKey, SecondKey, Data]):
    def __init__(self):
        super().__init__()
        self.partial_data: typing.Dict[AnyKey, typing.Dict[AnyKey, Data]] = dict()

    def __setitem__(self, key: FullKey, value: Data):
        frozen_key = frozenset(key)
        if len(frozen_key) != 2:
            raise ValueError("Using key that is not two distinct items in UnorderedPairDict")
        self.data[frozenset(key)] = value
        a, b = key
        if a not in self.partial_data:
            self.partial_data[a] = dict()
        if b not in self.partial_data:
            self.partial_data[b] = dict()
        self.partial_data[a][b] = value
        self.partial_data[b][a] = value

    def __getitem__(self, key: FullKey) -> Data:
        return self.data[frozenset(key)]

    def __delitem__(self, key: FullKey):
        frozen_key = frozenset(key)
        del self.data[frozen_key]
        a, b = key
        del self.partial_data[a][b]
        del self.partial_data[b][a]
        if not self.partial_data[a]:
            del self.partial_data[a]
        if not self.partial_data[b]:
            del self.partial_data[b]

    def get_partial_dict(self, key: AnyKey) -> typing.Dict[AnyKey, Data]:
        if key not in self.partial_data:
            return dict()
        return self.partial_data[key]


FirstKeyType = typing.TypeVar("FirstKeyType")
SecondKeyType = typing.TypeVar("SecondKeyType")
DataType = typing.TypeVar("DataType")
FirstKeyTypeOrSlice = typing.Union[FirstKeyType, slice]
SecondKeyTypeOrSlice = typing.Union[SecondKeyType, slice]


class TwoDimensionalDict(typing.Generic[FirstKeyType, SecondKeyType, DataType]):
    def __init__(self):
        self.data: typing.Dict[typing.Tuple[FirstKeyType, SecondKeyType], DataType] = dict()
        self.standard_ordered_data: typing.Dict[FirstKeyType, typing.Dict[SecondKeyType, DataType]] = dict()
        self.reverse_ordered_data: typing.Dict[SecondKeyType, typing.Dict[FirstKeyType, DataType]] = dict()

    def __getitem__(self, key: typing.Tuple[FirstKeyTypeOrSlice, SecondKeyTypeOrSlice])\
            -> typing.Union[DataType, typing.Dict[FirstKeyType, DataType], typing.Dict[SecondKeyType, DataType],
            typing.Dict[typing.Tuple[FirstKeyType, SecondKeyType], DataType]]:
        first_key, second_key = key
        if first_key == slice(None) and second_key == slice(None):
            return self.data
        elif first_key == slice(None):
            return self.reverse_ordered_data[second_key]
        elif second_key == slice(None):
            return self.standard_ordered_data[first_key]
        else:
            return self.data[key]

    def __setitem__(self, key: typing.Tuple[FirstKeyType, SecondKeyType], value: DataType):
        first_key, second_key = key
        if first_key is None or second_key is None:
            raise ValueError("Do you really want to assign with None key?")
        self.data[key] = value
        if first_key not in self.standard_ordered_data:
            self.standard_ordered_data[first_key] = dict()
        self.standard_ordered_data[first_key][second_key] = value
        if second_key not in self.reverse_ordered_data:
            self.reverse_ordered_data[second_key] = dict()
        self.reverse_ordered_data[second_key][first_key] = value

    def __delitem__(self, key: typing.Tuple[FirstKeyType, SecondKeyType]):
        first_key, second_key = key
        if first_key == slice(None) and second_key == slice(None):
            self.data = dict()
            self.standard_ordered_data = dict()
            self.reverse_ordered_data = dict()
        elif first_key == slice(None):
            deleted_first_keys = self.reverse_ordered_data[second_key].keys()
            del self.reverse_ordered_data[second_key]
            for deleted_first_key in deleted_first_keys:
                del self.data[(deleted_first_key, second_key)]
                del self.standard_ordered_data[deleted_first_key][second_key]
        elif second_key == slice(None):
            deleted_second_keys = self.standard_ordered_data[first_key].keys()
            del self.standard_ordered_data[first_key]
            for deleted_second_key in deleted_second_keys:
                del self.data[(first_key, deleted_second_key)]
                del self.reverse_ordered_data[deleted_second_key][first_key]
        else:
            first_key, second_key = key
            del self.data[key]
            del self.standard_ordered_data[first_key][second_key]
            del self.reverse_ordered_data[second_key][first_key]
            if not self.standard_ordered_data[first_key]:
                del self.standard_ordered_data[first_key]
            if not self.reverse_ordered_data[second_key]:
                del self.reverse_ordered_data[second_key]


def paged_query(query: sqlalchemy.orm.query, page_size: int):
    if query._offset is not None:
        raise RuntimeError("Paged query cannot evaluate query with set offset")
    if not query._order_by:
        raise RuntimeError("In order to split query into pages you need to order them by unique key")
    page_number = 0
    limit = query._limit
    items_to_reach_limit = (lambda: math.inf) if limit is None else (lambda: limit - page_size * page_number)
    this_page_query = query.limit(min(page_size, items_to_reach_limit()))
    while this_page_query.first():
        yield this_page_query
        page_number += 1
        if items_to_reach_limit() <= 0:
            break
        this_page_query = query.limit(min(page_size, items_to_reach_limit())).offset(page_number * page_size)


def smooth_paged_query(query: sqlalchemy.orm.query, page_size: int):
    for page in paged_query(query, page_size):
        for item in page:
            yield item


def get_logger(name: str, level: typing.Union[int, str, None] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger_stream_handler = logging.StreamHandler()
    logger_stream_handler.setLevel(0)
    logger_formatter = logging.Formatter('[{asctime}] {name} [{levelname}]: {message}', style='{')
    logger_stream_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_stream_handler)

    if level is not None:
        logger.setLevel(level)

    return logger


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--log', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str,
                          help='Threshold for logger level')


DEFAULT = object()


def shift_zip(iterable: typing.Iterable[typing.Any], prefix: typing.Any = DEFAULT, sufix: typing.Any = DEFAULT):
    iterator = iter(iterable)
    last = next(iterator)
    if prefix is not DEFAULT:
        yield prefix, last
    for item in iterator:
        yield last, item
        last = item
    if sufix is not DEFAULT:
        yield last, sufix


def print_query(query: sqlalchemy.orm.Query):
    print(query.statement.compile(compile_kwargs={'literal_binds': True}))


T = typing.TypeVar('T')


class PeekableIterator(typing.Iterator[T]):
    END = object()

    def __init__(self, iterable: typing.Iterable[T]):
        self.iterator: typing.Iterator[T] = iter(iterable)
        self.next_item: T = next(self.iterator, self.END)

    def __next__(self) -> T:
        if self.next_item is not self.END:
            result = self.next_item
            self.next_item = next(self.iterator, self.END)
            return result
        raise StopIteration

    def __iter__(self):
        return self

    def peek(self) -> T:
        return self.next_item


Item = typing.TypeVar("Item")
Aggregated = typing.TypeVar("Aggregated")


class GroupingIterator(typing.Generic[Item, Aggregated], typing.Iterator[Aggregated]):

    def __init__(self, iterable: typing.Iterable[Item], condition: typing.Callable[[Aggregated, Item], bool],
                 initialize: typing.Callable[[Item], Aggregated], append: typing.Callable[[Aggregated, Item], Aggregated]):
        self.iterator = PeekableIterator(iterable)
        self.condition = condition
        self.initialize = initialize
        self.append = append

    def __next__(self):
        aggregated: Aggregated = self.initialize(next(self.iterator))
        while self.iterator.peek() is not PeekableIterator.END and self.condition(aggregated, self.iterator.peek()):
            self.append(aggregated, next(self.iterator))
        return aggregated


class SimpleGroupingIterator(GroupingIterator[Item, typing.List[Item]]):

    def __init__(self, iterable: typing.Iterable[Item], condition: typing.Callable[[typing.List[Item], Item], bool]):
        super(SimpleGroupingIterator, self).__init__(iterable, condition, list, (lambda l, i: l.append(i)))


def intersperse_iterator(*args: typing.Iterable[T]) -> typing.Iterator[T]:
    memory: typing.Deque[typing.Iterator[T]] = collections.deque(map(iter, args))
    while memory:
        subiterator = memory.popleft()
        try:
            yield next(subiterator)
            memory.append(subiterator)
        except StopIteration:
            pass


def list_islice(list_: typing.List[T], *args: int):
    s = slice(*args)
    start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
    if start < 0 or stop - step < 0:
        raise RuntimeError("Negative values for start and (stop - step) is not implemented")
    it = iter(range(start, stop, step))
    for i in it:
        if i < 0:
            return None
        try:
            yield list_[i]
        except IndexError:
            return None


K = typing.TypeVar('K')
V = typing.TypeVar('V')


class ReactiveDefaultDict(typing.Dict[K, V]):
    def __init__(self, generator: typing.Callable[[K], V], *args, **kwargs):
        self.generator = generator
        super(ReactiveDefaultDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        self[key] = self.generator(key)
        return self[key]


def avg(l: typing.List[numbers.Real]) -> numbers.Real:
    return sum(l) / len(l)


def make_batches(base_iterable: typing.Iterable[T], batch_size: int) -> typing.Iterator[typing.List]:
    base_iterator = iter(base_iterable)
    next_batch = itertools.islice(base_iterator, batch_size)
    while next_batch:
        yield next_batch
        next_batch = itertools.islice(base_iterator, batch_size)


def parse_timedelta(input_string: str) -> datetime.timedelta:
    format_ = "%H:%M:%S.%f" if '.' in input_string else "%H:%M:%S"
    time = datetime.datetime.strptime(input_string, format_)
    time_delta = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond)
    return time_delta
