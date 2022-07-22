
"""

This file contains the position class, which is essentially a dataclass.
The functions are not methods of the class since Numba jitclasses aren't as fast and optimized as
normal jitted functions. The separation of functions and class also allows for caching of the
functions.

Functions in this file:

-flip_position
-is_attacked
-make_move
-undo_move
-make_capture
-undo_capture

"""

from move import *
from numba.experimental import jitclass


PIECE_MATCHER = np.array([
    'P',
    'N',
    'B',
    'R',
    'Q',
    'K',
])


# Numba's experimental Jitclasses require info on the attributes of the class
position_spec = [
    ("board", nb.int8[:]),  # Cannot be u-ints because subtraction of u-ints returns floats with numba
    ("own_king_position", nb.int8),
    ("opp_king_position", nb.int8),
    ("own_castle_ability", nb.boolean[:]),
    ("opp_castle_ability", nb.boolean[:]),
    ("ep_square", nb.int8),
    ("side", nb.int8),
    ("hash_key", nb.uint64)
]


@jitclass(position_spec)
class Position:
    def __init__(self):
        self.board = np.zeros(120, dtype=np.int8)
        self.own_king_position = 0
        self.opp_king_position = 0
        self.own_castle_ability = np.full(2, True)
        self.opp_castle_ability = np.full(2, True)
        self.ep_square = 0
        self.side = 0
        self.hash_key = 0

    # -- parses fen string to initialize position --
    def parse_fen(self, fen_string):

        fen_list = fen_string.strip().split()
        fen_board = fen_list[0]
        turn = fen_list[1]

        # -- boundaries for 12x10 mailbox --
        pos = 21
        for i in range(21):
            self.board[i] = PADDING

        # -- parse board --
        for i in fen_board:
            if i == "/":
                self.board[pos] = PADDING
                self.board[pos+1] = PADDING
                pos += 2
            elif i.isdigit():
                for j in range(ord(i) - 48):
                    self.board[pos] = EMPTY
                    pos += 1
            elif i.isalpha():
                idx = 0
                if i.islower():
                    idx = 6
                piece = i.upper()
                for j, p in enumerate(PIECE_MATCHER):
                    if piece == p:
                        idx += j
                self.board[pos] = idx
                if i == 'K':
                    self.own_king_position = pos
                elif i == 'k':
                    self.opp_king_position = pos
                pos += 1

        # -- boundaries for 12x10 mailbox --
        for i in range(21):
            self.board[pos+i] = PADDING

        for i in fen_list[2]:
            if i == "K":
                self.own_castle_ability[0] = True
            elif i == "Q":
                self.own_castle_ability[1] = True
            elif i == "k":
                self.opp_castle_ability[0] = True
            elif i == "q":
                self.opp_castle_ability[1] = True

        # -- en passant square --
        if len(fen_list[3]) > 1:
            square = [8 - (ord(fen_list[3][1]) - 48), ord(fen_list[3][0]) - 97]

            square = square[0] * 8 + square[1]

            square = square // 8 * 10 + 21 + square % 8

            self.ep_square = square
        else:
            self.ep_square = 0

        self.side = 0
        if turn == "b":
            flip_position(self)
            self.side = 1

        self.hash_key = compute_hash(self)

    def make_readable_board(self):
        new_board = " "
        for j, i in enumerate(self.board[21:100]):
            if (j + 1) % 10 == 0:
                new_board += "\n"
            if i == PADDING:
                new_board += " "
                continue
            if i == EMPTY:
                new_board += ". "
                continue
            idx = i
            piece = ""
            if idx == 0:
                piece = "\u265F "
            elif idx == 1:
                piece = "\u265E "
            elif idx == 2:
                piece = "\u265D "
            elif idx == 3:
                piece = "\u265C "
            elif idx == 4:
                piece = "\u265B "
            elif idx == 5:
                piece = "\u265A "
            elif idx == 6:
                piece = "\u2659 "
            elif idx == 7:
                piece = "\u2658 "
            elif idx == 8:
                piece = "\u2657 "
            elif idx == 9:
                piece = "\u2656 "
            elif idx == 10:
                piece = "\u2655 "
            elif idx == 11:
                piece = "\u2654 "
            new_board += piece

        new_board += "\n"
        return new_board


@nb.njit(nb.uint64(Position.class_type.instance_type))
def compute_hash(position):
    code = 0

    for i in range(64):
        pos = STANDARD_TO_MAILBOX[i]
        if position.board[pos] > 11:
            continue

        code ^= PIECE_HASH_KEYS[position.board[pos]][i]

    if position.ep_square:
        code ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]

    castle_bit = position.own_castle_ability[0] | position.own_castle_ability[1] << 1 | \
        position.opp_castle_ability[0] << 2 | position.opp_castle_ability[1] << 3

    code ^= CASTLE_HASH_KEYS[castle_bit]

    if position.side:  # side 1 is black, 0 is white
        code ^= SIDE_HASH_KEY

    return code


@nb.njit(cache=False)
def flip_position(position):
    position.board = np.flip(position.board)
    for i in range(64):
        pos = STANDARD_TO_MAILBOX[i]
        if position.board[pos] < 6:  # own piece
            position.board[pos] += 6
        elif position.board[pos] < 12:  # opponent's piece
            position.board[pos] -= 6

    temp0 = 119-position.own_king_position
    position.own_king_position = 119-position.opp_king_position
    position.opp_king_position = temp0

    temp0 = position.own_castle_ability[0]
    temp1 = position.own_castle_ability[1]
    position.own_castle_ability[0] = position.opp_castle_ability[0]
    position.own_castle_ability[1] = position.opp_castle_ability[1]
    position.opp_castle_ability[0] = temp0
    position.opp_castle_ability[1] = temp1


@nb.njit(cache=False)
def is_attacked(position, pos):
    board = position.board
    for piece in (4, 1):
        for increment in OPP_ATK_INCREMENTS[piece]:
            if increment == 0:
                break
            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == PADDING or occupied < 6:  # standing on own piece or outside of board
                    break

                if occupied < EMPTY:
                    if piece == occupied - 6:
                        return True

                    if piece == 1:  # if we are checking with knight and opponent piece is not knight
                        break
                    if occupied == 7:  # if we are checking with a queen and opponent piece is a knight
                        break
                    if occupied == 11:  # king
                        if new_pos == pos + increment:
                            return True
                        break

                    if occupied == 6:  # pawn
                        if new_pos == pos - 11 or \
                                new_pos == pos - 9:
                            return True
                        break

                    if occupied == 8:  # bishop
                        if increment in (-11, 11, 9, -9):
                            return True
                        break
                    if occupied == 9:  # rook
                        if increment in (-10, 1, 10, -1):
                            return True
                        break

                if piece == 1:  # if checking with knight
                    break
    return False


@nb.njit(cache=False)
def make_move(position, move):

    # Switch side
    position.side ^= 1
    position.hash_key ^= SIDE_HASH_KEY

    # Get move info
    castled_pos = np.array([-1, -1])
    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)
    occupied = get_occupied(move)
    move_type = get_move_type(move)

    # Normal move
    if move_type == 0:
        # Set the piece to the target square and hash it
        position.board[to_square] = selected
        position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[from_square]]

    # En passant move
    elif move_type == 1:
        # Set the piece to the target square and hash it
        position.board[to_square] = selected
        position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[from_square]]

        # Remove the en passant captured pawn and hash it
        position.board[to_square + 10] = EMPTY
        position.hash_key ^= PIECE_HASH_KEYS[6][MAILBOX_TO_STANDARD[to_square + 10]]

    # Castling move
    elif move_type == 2:
        # Set the piece to the target square and hash it
        position.board[to_square] = selected
        position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[from_square]]

        # Queen side castling
        if to_square < from_square:
            # Store the squares needed for legality checking
            castled_pos[0], castled_pos[1] = from_square, to_square + 1

            # Move the rook and hash it
            position.board[from_square - 1] = 3
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[from_square - 1]]

            # Remove the rook from the source square and hash it
            position.board[91] = EMPTY
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[91]]

        # King side castling
        else:
            # Store the squares needed for legality checking
            castled_pos[0], castled_pos[1] = from_square, to_square - 1

            # Move the rook and hash it
            position.board[from_square + 1] = 3
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[from_square + 1]]

            # Remove the rook from the source square and hash it
            position.board[98] = EMPTY
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[98]]

    # Promotion move
    elif move_type == 3:
        # Get the promoted piece, set it in the new location, and hash it
        promoted_piece = get_promotion_piece(move)
        position.board[to_square] = promoted_piece
        position.hash_key ^= PIECE_HASH_KEYS[promoted_piece][MAILBOX_TO_STANDARD[from_square]]

    # Remove the piece from the source square
    position.board[from_square] = EMPTY
    position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[to_square]]

    # Hash out the occupied piece if it is a capture
    if get_is_capture(move):
        position.hash_key ^= PIECE_HASH_KEYS[occupied][MAILBOX_TO_STANDARD[to_square]]

    # Change the king position for check detection
    if selected == 5:
        position.own_king_position = to_square

    # Legal move checking.
    # Return False if we are in check after our move or castling isn't legal.
    if is_attacked(position, position.own_king_position):
        return False
    elif castled_pos[0] != -1:
        if is_attacked(position, castled_pos[0]):
            return False
        elif is_attacked(position, castled_pos[1]):
            return False

    # --- The move is legal ---

    # Double pawn push
    if selected == 0 and to_square - from_square == -20:
        position.ep_square = 109 - to_square  # 119 - (to_square + 10)
        position.hash_key ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]  # Set new EP hash
    else:
        position.hash_key ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]  # Remove EP hash
        position.ep_square = 0

    # We first reset the castling right hash here
    castle_bit = position.own_castle_ability[0] | position.own_castle_ability[1] << 1 | \
                 position.opp_castle_ability[0] << 2 | position.opp_castle_ability[1] << 3
    position.hash_key ^= CASTLE_HASH_KEYS[castle_bit]

    # Then we update the castling rights if necessary
    if selected == 5:
        position.own_castle_ability[0] = False
        position.own_castle_ability[1] = False

    if from_square == A1:
        position.own_castle_ability[0] = False
    elif from_square == H1:
        position.own_castle_ability[1] = False
    if to_square == A8:
        position.opp_castle_ability[1] = False
    elif to_square == H8:
        position.opp_castle_ability[0] = False

    # After that we re-add the castling right hash
    castle_bit = position.own_castle_ability[0] | position.own_castle_ability[1] << 1 | \
                 position.opp_castle_ability[0] << 2 | position.opp_castle_ability[1] << 3
    position.hash_key ^= CASTLE_HASH_KEYS[castle_bit]

    return True


@nb.njit(cache=False)
def undo_move(position, move, current_own_castle_ability, current_opp_castle_ability, current_ep):

    # Switch side
    position.side ^= 1
    position.hash_key ^= SIDE_HASH_KEY

    # Get move info
    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)
    occupied = get_occupied(move)
    move_type = get_move_type(move)

    # En Passant move
    if move_type == 1:
        # Place the en passant captured pawn back and hash it
        position.board[to_square + 10] = 6
        position.hash_key ^= PIECE_HASH_KEYS[6][MAILBOX_TO_STANDARD[to_square + 10]]

    # Castling move
    if move_type == 2:
        # Queen side castle
        if to_square < from_square:
            # Move the rook back and hash it
            position.board[91] = 3
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[91]]

            # Remove the rook from the destination square and hash it
            position.board[from_square - 1] = EMPTY
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[from_square - 1]]

        # King side castle
        else:
            # Move the rook back and hash it
            position.board[98] = 3
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[98]]

            # Remove the rook from the destination square and hash it
            position.board[from_square + 1] = EMPTY
            position.hash_key ^= PIECE_HASH_KEYS[3][MAILBOX_TO_STANDARD[from_square + 1]]

    # Place occupied piece/value back in the destination square and hash it
    # Set the source square back to the selected piece and hash it
    position.hash_key ^= PIECE_HASH_KEYS[position.board[to_square]][MAILBOX_TO_STANDARD[to_square]]
    position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[from_square]]

    position.board[to_square] = occupied
    position.board[from_square] = selected

    # If the move is a capture then un-hash the captured piece
    if get_is_capture(move):
        position.hash_key ^= PIECE_HASH_KEYS[occupied][MAILBOX_TO_STANDARD[to_square]]

    # The en passant square has changed
    if position.ep_square != current_ep:
        position.hash_key ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]
        position.ep_square = current_ep

    # We first reset the castling right hash here
    castle_bit = position.own_castle_ability[0] | position.own_castle_ability[1] << 1 | \
                 position.opp_castle_ability[0] << 2 | position.opp_castle_ability[1] << 3
    position.hash_key ^= CASTLE_HASH_KEYS[castle_bit]

    # Then we revert the castling rights
    position.own_castle_ability[0] = current_own_castle_ability[0]
    position.own_castle_ability[1] = current_own_castle_ability[1]

    position.opp_castle_ability[0] = current_opp_castle_ability[0]
    position.opp_castle_ability[1] = current_opp_castle_ability[1]

    # After that we re-add the castling right hash
    castle_bit = position.own_castle_ability[0] | position.own_castle_ability[1] << 1 | \
                 position.opp_castle_ability[0] << 2 | position.opp_castle_ability[1] << 3
    position.hash_key ^= CASTLE_HASH_KEYS[castle_bit]

    # Reset the king position if it has moved
    if selected == 5:
        position.own_king_position = from_square


@nb.njit(cache=False)
def make_capture(position, move):

    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)

    position.board[to_square] = selected
    position.board[from_square] = EMPTY

    if selected == 5:
        position.own_king_position = to_square

    if is_attacked(position, position.own_king_position):
        return False

    return True


@nb.njit(cache=False)
def undo_capture(position, move):

    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)
    occupied = get_occupied(move)

    position.board[to_square] = occupied
    position.board[from_square] = selected

    if selected == 5:
        position.own_king_position = from_square



'''
incheck njit
(1, 0, 0, 0, 0) 3.624999999729539e-06
(20, 0, 0, 0, 0) 0.5131300830000001
(400, 0, 0, 0, 0) 0.010259709000001394
(8902, 34, 0, 12, 0) 0.25145966700000066

3: (1, 0, 0, 0, 0) 3.9590000000533365e-06
3: (20, 0, 0, 0, 0) 0.9135626250000001
3: (400, 0, 0, 0, 0) 0.02379966599999994
3: (8902, 34, 0, 12, 0) 0.5183446670000003

Makemove njit
(1, 0, 0, 0, 0) 1.3330000001587905e-06
(20, 0, 0, 0, 0) 0.8349006249999995
(400, 0, 0, 0, 0) 0.011415915999999804
(8902, 34, 0, 12, 0) 0.26993779199999945

3: (1, 0, 0, 0, 0) 3.250000000010189e-06
3: (20, 0, 0, 0, 0) 1.092408416
3: (400, 0, 0, 0, 0) 0.4462142920000003
3: (8902, 34, 0, 12, 0) 0.2711170829999996

Undomove njit
(1, 0, 0, 0, 0) 1.708000000100185e-06
(20, 0, 0, 0, 0) 1.0710964580000004
(400, 0, 0, 0, 0) 0.01103325000000055
(8902, 34, 0, 12, 0) 0.23741866700000003

3: (1, 0, 0, 0, 0) 2.8749999999577724e-06
3: (20, 0, 0, 0, 0) 1.192687125
3: (400, 0, 0, 0, 0) 0.47335604200000003
3: (8902, 34, 0, 12, 0) 0.27404020900000026

Pseudo Legal Move
(1, 0, 0, 0, 0) 9.58000000217396e-07
(20, 0, 0, 0, 0) 1.3550063749999994
(400, 0, 0, 0, 0) 0.002123750000000868
(8902, 34, 0, 12, 0) 0.04826562500000087

3: (1, 0, 0, 0, 0) 3.749999999969056e-06
3: (20, 0, 0, 0, 0) 1.6368319580000001
3: (400, 0, 0, 0, 0) 0.782766375
3: (8902, 34, 0, 12, 0) 0.23104925000000032

Perft
(1, 0, 0, 0, 0) 1.6898107920000003
(20, 0, 0, 0, 0) 0.0004498330000011208
(400, 0, 0, 0, 0) 0.0002648749999991651
(8902, 34, 0, 12, 0) 0.005955042000000077

3: (1, 0, 0, 0, 0) 2.778665666
3: (20, 0, 0, 0, 0) 0.00040354200000036755
3: (400, 0, 0, 0, 0) 0.00028866599999988196
3: (8902, 34, 0, 12, 0) 0.0043427080000002505

--
(1, 0, 0, 0, 0) 1.615739875000001
(20, 0, 0, 0, 0) 0.0004347500000001503
(400, 0, 0, 0, 0) 0.0002569169999997456
(8902, 34, 0, 12, 0) 0.005752792000000895
(197281, 1576, 0, 469, 461) 0.12309824999999996
(4865609, 82719, 258, 27351, 15375) 3.134478208000001

3: (1, 0, 0, 0, 0) 2.963718333
3: (20, 0, 0, 0, 0) 0.0004398749999996454
3: (400, 0, 0, 0, 0) 0.00031437499999986684
3: (8902, 34, 0, 12, 0) 0.004687374999999605
3: (197281, 1576, 0, 469, 461) 0.14667049999999993
3: (4865609, 82719, 258, 27351, 15375) 2.412793375
'''


