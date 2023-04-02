from csv import reader

from torch import tensor, float64

from named_tensor import NamedTensor


class CSVNumbersReader:
    def __init__(self, fname: str):
        self.f = open(fname, 'r', newline='')
        self.rows = reader(self.f, delimiter=',', quotechar='"')
        self.header = tuple(next(self.rows))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def read(self, n=1024):
        rows_float = []
        try:
            for _ in range(n):
                rows_float.append(tuple(float(x) for x in next(self.rows)))
        except StopIteration:
            pass
        return NamedTensor(
            tensor(rows_float, dtype=float64), axis_descr=self.header
        )
