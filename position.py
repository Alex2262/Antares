
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
from position_class import Position


PIECE_MATCHER = np.array((
    'P',
    'N',
    'B',
    'R',
    'Q',
    'K',
))


@nb.njit(nb.uint64(Position.class_type.instance_type), cache=True)
def compute_hash(position):
    code = 0

    for i in range(64):
        pos = STANDARD_TO_MAILBOX[i]
        if position.board[pos] > BLACK_KING:
            continue

        code ^= PIECE_HASH_KEYS[position.board[pos]][i]

    if position.ep_square:
        code ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]

    code ^= CASTLE_HASH_KEYS[position.castle_ability_bits]

    if position.side:  # side 1 is black, 0 is white
        code ^= SIDE_HASH_KEY

    return code


# @nb.njit(nb.boolean(Position.class_type.instance_type, nb.int8), cache=True)
@nb.njit
def is_attacked(position, pos):
    board = position.board
    if position.side == 0:
        for piece in (WHITE_QUEEN, WHITE_KNIGHT):
            for increment in BLACK_ATK_INCREMENTS[piece]:
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

                        if piece == WHITE_KNIGHT:  # if we are checking with knight and opponent piece is not knight
                            break
                        if occupied == BLACK_KNIGHT:  # if we are checking with a queen and opponent piece is a knight
                            break
                        if occupied == BLACK_KING:  # king
                            if new_pos == pos + increment:
                                return True
                            break

                        if occupied == BLACK_PAWN:  # pawn
                            if new_pos == pos - 11 or \
                                    new_pos == pos - 9:
                                return True
                            break

                        if occupied == BLACK_BISHOP:  # bishop
                            if increment in (-11, 11, 9, -9):
                                return True
                            break
                        if occupied == BLACK_ROOK:  # rook
                            if increment in (-10, 1, 10, -1):
                                return True
                            break

                    if piece == WHITE_KNIGHT:  # if checking with knight
                        break
    else:
        for piece in (BLACK_QUEEN, BLACK_KNIGHT):
            for increment in WHITE_ATK_INCREMENTS[piece-BLACK_PAWN]:
                if increment == 0:
                    break
                new_pos = pos
                while True:
                    new_pos += increment
                    occupied = board[new_pos]
                    if occupied != EMPTY and occupied > WHITE_KING:  # standing on own piece or outside of board
                        break

                    if occupied < BLACK_PAWN:
                        if piece == occupied + BLACK_PAWN:
                            return True

                        if piece == BLACK_KNIGHT:  # if we are checking with knight and opponent piece is not knight
                            break
                        if occupied == WHITE_KNIGHT:  # if we are checking with a queen and opponent piece is a knight
                            break
                        if occupied == WHITE_KING:  # king
                            if new_pos == pos + increment:
                                return True
                            break

                        if occupied == WHITE_PAWN:  # pawn
                            if new_pos == pos + 11 or \
                                    new_pos == pos + 9:
                                return True
                            break

                        if occupied == WHITE_BISHOP:  # bishop
                            if increment in (-11, 11, 9, -9):
                                return True
                            break
                        if occupied == WHITE_ROOK:  # rook
                            if increment in (-10, 1, 10, -1):
                                return True
                            break

                    if piece == BLACK_KNIGHT:  # if checking with knight
                        break

    return False


# @nb.njit(nb.boolean(Position.class_type.instance_type, MOVE_TYPE), cache=True)
@nb.njit
def make_move(position, move):

    # Get move info
    castled_pos = np.array([0, 0])
    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)
    occupied = get_occupied(move)
    move_type = get_move_type(move)

    # Normal move
    if move_type == MOVE_TYPE_NORMAL:
        # Set the piece to the target square and hash it
        position.board[to_square] = selected
        position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[to_square]]

    # En passant move
    elif move_type == MOVE_TYPE_EP:
        # Set the piece to the target square and hash it
        position.board[to_square] = selected
        position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[to_square]]

        # Remove the en passant captured pawn and hash it
        # If white to move, to_square - (0*20) + 10 == to_square + 10
        # If black to move, to_square - (1*20) + 10 == to_square - 10
        position.board[to_square - position.side * 20 + 10] = EMPTY
        position.hash_key ^= PIECE_HASH_KEYS[
            BLACK_PAWN - position.side * BLACK_PAWN][MAILBOX_TO_STANDARD[to_square - position.side * 20 + 10]]

    # Castling move
    elif move_type == MOVE_TYPE_CASTLE:
        # Set the piece to the target square and hash it
        position.board[to_square] = selected
        position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[to_square]]

        # Queen side castling
        if to_square < from_square:
            castled_pos[0], castled_pos[1] = to_square - 2, to_square + 1  # A1/A8, D1/D8
        # King side castling
        else:
            castled_pos[0], castled_pos[1] = to_square + 1, to_square - 1  # H1/H8, F1/F8

        # White Rook + position.side * 6 will calculate which side's rook will be used.
        # Move the rook and hash it
        position.board[castled_pos[1]] = WHITE_ROOK + position.side * BLACK_PAWN
        position.hash_key ^= PIECE_HASH_KEYS[WHITE_ROOK + position.side * BLACK_PAWN][
            MAILBOX_TO_STANDARD[castled_pos[1]]]

        # Remove the rook from the source square and hash it
        position.board[castled_pos[0]] = EMPTY
        position.hash_key ^= PIECE_HASH_KEYS[WHITE_ROOK + position.side * BLACK_PAWN][
            MAILBOX_TO_STANDARD[castled_pos[0]]]

    # Promotion move
    elif move_type == MOVE_TYPE_PROMOTION:
        # Get the promoted piece, set it in the new location, and hash it
        promoted_piece = get_promotion_piece(move)
        position.board[to_square] = promoted_piece
        position.hash_key ^= PIECE_HASH_KEYS[promoted_piece][MAILBOX_TO_STANDARD[to_square]]

    # Remove the piece from the source square
    position.board[from_square] = EMPTY
    position.hash_key ^= PIECE_HASH_KEYS[selected][MAILBOX_TO_STANDARD[from_square]]

    # Hash out the occupied piece if it is a capture
    if get_is_capture(move):
        position.hash_key ^= PIECE_HASH_KEYS[occupied][MAILBOX_TO_STANDARD[to_square]]

    # Change the king position for check detection
    if selected == WHITE_KING or selected == BLACK_KING:
        position.king_positions[position.side] = to_square

    # Legal move checking.
    # Return False if we are in check after our move or castling isn't legal.
    if is_attacked(position, position.king_positions[position.side]):
        return False
    elif castled_pos[0]:
        # If we have castled, then we already checked to_square with is_attacked since the king moved.
        # We then check the position of where the rook would be, and also where the king originally was
        if is_attacked(position, castled_pos[1]):
            return False
        elif is_attacked(position, from_square):
            return False

    # --- The move is legal ---

    # Double pawn push
    if (selected == WHITE_PAWN or selected == BLACK_PAWN) and abs(to_square - from_square) == 20:
        if position.ep_square:
            position.hash_key ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]

        position.ep_square = to_square - position.side * 20 + 10  # 119 - (to_square + 10)
        position.hash_key ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]  # Set new EP hash

    # Reset ep square since it is not a double pawn push
    else:
        if position.ep_square:
            position.hash_key ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]  # Remove EP hash
            position.ep_square = 0

    # We first reset the castling right hash here
    position.hash_key ^= CASTLE_HASH_KEYS[position.castle_ability_bits]

    if selected == WHITE_KING:
        position.castle_ability_bits &= ~(1 << 0)
        position.castle_ability_bits &= ~(1 << 1)
    elif selected == BLACK_KING:
        position.castle_ability_bits &= ~(1 << 2)
        position.castle_ability_bits &= ~(1 << 3)

    # Update the castling rights if necessary
    if from_square == H1:
        position.castle_ability_bits &= ~(1 << 0)
    elif from_square == A1:
        position.castle_ability_bits &= ~(1 << 1)
    if to_square == H8:
        position.castle_ability_bits &= ~(1 << 2)
    elif to_square == A8:
        position.castle_ability_bits &= ~(1 << 3)

    # After that we re-add the castling right hash
    position.hash_key ^= CASTLE_HASH_KEYS[position.castle_ability_bits]

    # Switch hash side (actual side is switched in loop)
    position.hash_key ^= SIDE_HASH_KEY

    return True


# @nb.njit(nb.void(Position.class_type.instance_type, MOVE_TYPE, nb.int8, nb.uint8, nb.uint64), cache=True)
@nb.njit
def undo_move(position, move, current_ep, current_castle_ability_bits, current_hash_key):

    # Restore hash
    position.hash_key = current_hash_key

    # Get move info
    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)
    occupied = get_occupied(move)
    move_type = get_move_type(move)

    # En Passant move
    if move_type == MOVE_TYPE_EP:
        # Place the en passant captured pawn back and hash it
        position.board[to_square - position.side * 20 + 10] = BLACK_PAWN - position.side * BLACK_PAWN

    # Castling move
    if move_type == MOVE_TYPE_CASTLE:
        # Queen side castle
        if to_square < from_square:
            # Move the rook back
            position.board[to_square - 2] = WHITE_ROOK + position.side * BLACK_PAWN

            # Remove the rook from the destination square
            position.board[from_square - 1] = EMPTY

        # King side castle
        else:
            # Move the rook back
            position.board[to_square + 1] = WHITE_ROOK + position.side * BLACK_PAWN

            # Remove the rook from the destination square
            position.board[from_square + 1] = EMPTY

    # Place occupied piece/value back in the destination square
    # Set the source square back to the selected piece
    position.board[to_square] = occupied
    position.board[from_square] = selected

    # The en passant square has changed
    if position.ep_square != current_ep:
        position.ep_square = current_ep

    # Revert the castling rights
    position.castle_ability_bits = current_castle_ability_bits

    # Reset the king position if it has moved
    if selected == WHITE_KING or selected == BLACK_KING:
        position.king_positions[position.side] = from_square


# @nb.njit(nb.boolean(Position.class_type.instance_type, MOVE_TYPE), cache=True)
@nb.njit
def make_capture(position, move):

    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)

    position.board[to_square] = selected
    position.board[from_square] = EMPTY

    if selected == WHITE_KING or selected == BLACK_KING:
        position.king_positions[position.side] = to_square

    if is_attacked(position, position.king_positions[position.side]):
        return False

    return True


# @nb.njit(nb.void(Position.class_type.instance_type, MOVE_TYPE), cache=True)
@nb.njit
def undo_capture(position, move):

    from_square = get_from_square(move)
    to_square = get_to_square(move)
    selected = get_selected(move)
    occupied = get_occupied(move)

    position.board[to_square] = occupied
    position.board[from_square] = selected

    if selected == WHITE_KING or selected == BLACK_KING:
        position.king_positions[position.side] = from_square


# @nb.njit(nb.void(Position.class_type.instance_type), cache=True)
@nb.njit
def make_null_move(position):

    position.side ^= 1
    position.hash_key ^= SIDE_HASH_KEY

    if position.ep_square:
        position.hash_key ^= EP_HASH_KEYS[MAILBOX_TO_STANDARD[position.ep_square]]
        position.ep_square = 0


# @nb.njit(nb.void(Position.class_type.instance_type, nb.int8, nb.uint64), cache=True)
@nb.njit
def undo_null_move(position, current_ep, current_hash_key):

    position.side ^= 1
    position.ep_square = current_ep
    position.hash_key = current_hash_key


# @nb.njit(nb.void(Position.class_type.instance_type, nb.types.unicode_type), cache=True)
@nb.njit
def parse_fen(position, fen_string):
    fen_list = fen_string.strip().split()
    fen_board = fen_list[0]
    turn = fen_list[1]

    # -- boundaries for 12x10 mailbox --
    pos = 21
    for i in range(21):
        position.board[i] = PADDING

    # -- parse board --
    for i in fen_board:
        if i == "/":
            position.board[pos] = PADDING
            position.board[pos + 1] = PADDING
            pos += 2
        elif i.isdigit():
            for j in range(ord(i) - 48):
                position.board[pos] = EMPTY
                pos += 1
        elif i.isalpha():
            idx = 0
            if i.islower():
                idx = 6
            piece = i.upper()
            for j, p in enumerate(PIECE_MATCHER):
                if piece == p:
                    idx += j
            position.board[pos] = idx
            if i == 'K':
                position.king_positions[0] = pos
            elif i == 'k':
                position.king_positions[1] = pos
            pos += 1

    # -- boundaries for 12x10 mailbox --
    for i in range(21):
        position.board[pos + i] = PADDING

    position.castle_ability_bits = 0
    for i in fen_list[2]:
        if i == "K":
            position.castle_ability_bits |= 1
        elif i == "Q":
            position.castle_ability_bits |= 2
        elif i == "k":
            position.castle_ability_bits |= 4
        elif i == "q":
            position.castle_ability_bits |= 8

    # -- en passant square --
    if len(fen_list[3]) > 1:
        square = [8 - (ord(fen_list[3][1]) - 48), ord(fen_list[3][0]) - 97]

        square = square[0] * 8 + square[1]

        position.ep_square = STANDARD_TO_MAILBOX[square]
    else:
        position.ep_square = 0

    position.side = 0
    if turn == "b":
        position.side = 1

    position.hash_key = compute_hash(position)


@nb.njit(nb.types.unicode_type(Position.class_type.instance_type), cache=True)
def make_readable_board(position):
    new_board = " "
    for j, i in enumerate(position.board[21:100]):
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
