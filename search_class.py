
from numba.experimental import jitclass
from utilities import *

search_spec = [
    ("max_depth", nb.uint16),
    ("max_qdepth", nb.uint16),
    ("max_time", nb.double),        # seconds
    ("min_depth", nb.uint16),
    ("current_search_depth", nb.uint16),
    ("ply", nb.uint16),             # opposite of depth counter
    ("start_time", nb.uint64),
    ("node_count", nb.uint64),
    ("pv_table", nb.uint64[:, :]),  # implementation of pv and pv scoring comes from TSCP engine
    ("pv_length", nb.uint64[:]),
    ("killer_moves", nb.uint64[:, :]),
    ("history_moves", nb.uint64[:, :]),
    ("transposition_table", NUMBA_HASH_TYPE[:]),
    ("stopped", nb.boolean)

]


@jitclass(spec=search_spec)
class Search:

    def __init__(self):
        # self.TRANSPOSITION_TABLE = {}
        # self.WIN_TABLE = {}

        self.max_depth = 30
        self.max_qdepth = 4
        self.min_depth = 2
        self.current_search_depth = 0
        self.ply = 0

        self.max_time = 10
        self.start_time = 0

        self.node_count = 0

        self.pv_table = np.zeros((self.max_depth, self.max_depth), dtype=np.uint64)
        self.pv_length = np.zeros(self.max_depth, dtype=np.uint64)

        # Killer moves [id][ply]
        self.killer_moves = np.zeros((2, self.max_depth), dtype=np.uint64)
        # History moves [piece][square]
        self.history_moves = np.zeros((12, 64), dtype=np.uint64)

        self.transposition_table = np.zeros(MAX_HASH_SIZE, dtype=NUMBA_HASH_TYPE)

        self.stopped = False

        # self.aspiration_window = 65  # in centi pawns

