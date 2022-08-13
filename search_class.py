

from utilities import *
from numba import njit
from numba.core import types
from numba.experimental import jitclass
from numba.experimental import structref

search_spec = [
    ("max_depth", nb.uint16),
    ("max_qdepth", nb.uint16),
    ("max_time", nb.uint64),        # seconds
    ("min_depth", nb.uint16),
    ("current_search_depth", nb.uint16),
    ("ply", nb.uint16),             # opposite of depth counter
    ("start_time", nb.double),
    ("node_count", nb.uint64),
    ("killer_moves", MOVE_TYPE[:, :]),
    ("history_moves", nb.uint32[:, :]),
    ("transposition_table", NUMBA_HASH_TYPE[:]),
    ("repetition_table", nb.uint64[:]),
    ("repetition_index", nb.uint16),
    ("stopped", nb.boolean)

]


@jitclass(spec=search_spec)
class Search:

    def __init__(self):

        self.max_depth = 64
        self.max_qdepth = 1000
        self.min_depth = 2
        self.current_search_depth = 0
        self.ply = 0

        self.max_time = 10000
        self.start_time = 0

        self.node_count = 0

        # Killer moves [id][ply]
        self.killer_moves = np.zeros((2, self.max_depth), dtype=np.uint32)
        # History moves [piece][square]
        self.history_moves = np.zeros((12, 64), dtype=np.uint32)

        self.transposition_table = np.zeros(MAX_HASH_SIZE, dtype=NUMBA_HASH_TYPE)

        self.repetition_table = np.zeros(REPETITION_TABLE_SIZE, dtype=np.uint64)
        self.repetition_index = 0

        self.stopped = False

        # self.aspiration_window = 65  # in centi pawns


@structref.register
class SearchStructType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class SearchStruct(structref.StructRefProxy):
    def __new__(cls, max_depth, max_qdepth, min_depth,
                current_search_depth, ply, max_time, start_time, node_count,
                killer_moves, history_moves, transposition_table, repetition_table,
                repetition_index, stopped):

        return structref.StructRefProxy.__new__(cls, max_depth, max_qdepth, min_depth,
                current_search_depth, ply, max_time, start_time, node_count,
                killer_moves, history_moves, transposition_table, repetition_table,
                repetition_index, stopped)

    @property
    def max_depth(self):
        return SearchStruct_get_max_depth(self)

    @property
    def max_qdepth(self):
        return SearchStruct_get_max_qdepth(self)

    @property
    def min_depth(self):
        return SearchStruct_get_min_depth(self)

    @property
    def current_search_depth(self):
        return SearchStruct_get_current_search_depth(self)

    @property
    def ply(self):
        return SearchStruct_get_ply(self)

    @property
    def max_time(self):
        return SearchStruct_get_max_time(self)

    @property
    def start_time(self):
        return SearchStruct_get_start_time(self)

    @property
    def node_count(self):
        return SearchStruct_get_node_count(self)

    @property
    def killer_moves(self):
        return SearchStruct_get_killer_moves(self)

    @property
    def history_moves(self):
        return SearchStruct_get_history_moves(self)

    @property
    def transposition_table(self):
        return SearchStruct_get_transposition_table(self)

    @property
    def repetition_table(self):
        return SearchStruct_get_repetition_table(self)

    @property
    def repetition_index(self):
        return SearchStruct_get_repetition_index(self)

    @property
    def stopped(self):
        return SearchStruct_get_stopped(self)


@njit
def SearchStruct_get_max_depth(self):
    return self.max_depth


@njit
def SearchStruct_get_max_qdepth(self):
    return self.max_qdepth


@njit
def SearchStruct_get_min_depth(self):
    return self.min_depth


@njit
def SearchStruct_get_current_search_depth(self):
    return self.current_search_depth


@njit
def SearchStruct_get_ply(self):
    return self.ply


@njit
def SearchStruct_get_max_time(self):
    return self.max_time


@njit
def SearchStruct_get_start_time(self):
    return self.start_time


@njit
def SearchStruct_get_node_count(self):
    return self.node_count


@njit
def SearchStruct_get_killer_moves(self):
    return self.killer_moves


@njit
def SearchStruct_get_history_moves(self):
    return self.history_moves


@njit
def SearchStruct_get_transposition_table(self):
    return self.transposition_table


@njit
def SearchStruct_get_repetition_table(self):
    return self.repetition_table


@njit
def SearchStruct_get_repetition_index(self):
    return self.repetition_index


@njit
def SearchStruct_get_stopped(self):
    return self.stopped


@njit
def SearchStruct_set_max_time(engine, t):
    engine.max_time = t


@njit
def SearchStruct_set_max_depth(engine, d):
    engine.max_depth = d


@njit
def SearchStruct_set_repetition_index(engine, r):
    engine.repetition_index = r


structref.define_proxy(SearchStruct, SearchStructType, ["max_depth", "max_qdepth", "min_depth",
                "current_search_depth", "ply", "max_time", "start_time", "node_count",
                "killer_moves", "history_moves", "transposition_table", "repetition_table",
                "repetition_index", "stopped"])


@njit(cache=True)
def init_search():
    engine = SearchStruct(max_depth=nb.uint16(64),
                          max_qdepth=nb.uint16(1000),
                          min_depth=nb.uint16(2),
                          current_search_depth=nb.int16(0),
                          ply=nb.int16(0),
                          max_time=nb.uint64(10000),
                          start_time=nb.double(0),
                          node_count=nb.uint64(0),
                          killer_moves=np.zeros((2, 64), dtype=np.uint32),
                          history_moves=np.zeros((12, 64), dtype=np.uint32),
                          transposition_table=np.zeros(MAX_HASH_SIZE, dtype=NUMBA_HASH_TYPE),
                          repetition_table=np.zeros(REPETITION_TABLE_SIZE, dtype=np.uint64),
                          repetition_index=nb.uint16(0),
                          stopped=False
                          )

    return engine
