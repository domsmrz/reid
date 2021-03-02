'''Implementation of standard UnionFind structure'''


import typing


DataType = typing.TypeVar("DataType", bound=typing.Hashable)
InnerDataType = typing.TypeVar("InnerDataType", bound=typing.Hashable)


class UnionFind(typing.Generic[DataType]):
    class Element(typing.Generic[InnerDataType]):
        def __init__(self, structure: 'UnionFind[InnerDataType]', data: DataType):
            self._roots: typing.Set['UnionFind.Element[InnerDataType]'] = structure.roots
            self.data: InnerDataType = data
            self.rank: int = 0
            self._parent: typing.Optional['UnionFind.Element[InnerDataType]'] = None
            self._children: typing.Set['UnionFind.Element[InnerDataType]'] = set()

        @property
        def parent(self) -> 'UnionFind.Element[InnerDataType]':
            return self._parent

        @parent.setter
        def parent(self, parent: 'UnionFind.Element[InnerDataType]'):
            if self._parent is not None:
                self._parent._children.remove(self)
            elif parent is not None:
                self._roots.remove(self)

            self._parent = parent
            if parent is not None:
                self._parent._children.add(self)

        @property
        def children(self) -> typing.Iterator['UnionFind.Element[InnerDataType]']:
            return iter(self._children)

        def get_subtree(self) -> typing.Iterator['UnionFind.Element[InnerDataType]']:
            yield self
            for child in self._children:
                yield from child.get_subtree()

    def __init__(self, initial_items: typing.Optional[typing.Iterable[DataType]] = None):
        self.roots: typing.Set['UnionFind.Element[DataType]'] = set()
        self.elements: typing.Dict[DataType, UnionFind.Element[DataType]] = dict()
        if initial_items is not None:
            for item in initial_items:
                self.add(item)

    def add(self, data: DataType) -> None:
        element: UnionFind.Element[DataType] = self.Element(structure=self, data=data)
        self.roots.add(element)
        self.elements[data] = element

    @staticmethod
    def find_element(element: 'UnionFind.Element[DataType]') -> 'UnionFind.Element[DataType]':
        path = list()
        while element.parent is not None:
            path.append(element)
            element = element.parent
        root = element
        for element in path:
            element.parent = root
        return root

    def find(self, item: DataType) -> DataType:
        element = self.elements[item]
        root_element = self.find_element(element)
        return root_element.data

    def union_elements(self, element_a: 'UnionFind.Element[DataType]', element_b: 'UnionFind.Element[DataType]')\
            -> 'UnionFind.Element[DataType]':
        root_a = self.find_element(element_a)
        root_b = self.find_element(element_b)

        if root_a is root_b:
            return root_a

        if root_a.rank < root_b.rank:
            root_a, root_b = root_b, root_a

        root_b.parent = root_a
        if root_a.rank == root_b.rank:
            root_a.rank += 1
        return root_a

    def union(self, item_a: DataType, item_b: DataType) -> DataType:
        element_a = self.elements[item_a]
        element_b = self.elements[item_b]
        return self.union_elements(element_a, element_b).data

    def get_sets(self) -> typing.Iterator[typing.Iterator[DataType]]:
        for root in self.roots:
            yield (node.data for node in root.get_subtree())
