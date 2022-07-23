

from evaluation import score_move, score_capture
from move import *


@nb.njit
def get_pseudo_legal_moves(position):
    moves = []
    board = position.board
    for i in range(64):
        pos = STANDARD_TO_MAILBOX[i]
        piece = board[pos]
        if piece > 5:  # if not own piece
            continue

        for increment in OWN_INCREMENTS[piece]:
            if increment == 0:
                break

            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == PADDING or occupied < 6:  # standing on own piece or outside of board
                    break
                elif piece == 0 and increment in (-10, -20) and board[pos-10] != EMPTY:
                    break
                elif piece == 0 and increment == -20 and (pos < 81 or occupied != EMPTY):
                    break

                if piece == 0 and increment in (-11, -9) and occupied == EMPTY:
                    if new_pos == position.ep_square:
                        moves.append(encode_move(pos, new_pos,
                                                 piece, EMPTY,
                                                 1, 0))
                    break
                elif piece == 0 and new_pos < 31:
                    for j in range(1, 5):
                        moves.append(encode_move(pos, new_pos,
                                                 piece, occupied,
                                                 3, j))
                    break
                moves.append(encode_move(pos, new_pos,
                                         piece, occupied,
                                         0, 0))

                if piece in (0, 1, 5) or occupied < EMPTY:  # if is pawn, knight, king or opposing piece
                    break

                if pos == A1 and board[new_pos+1] == 5 and position.own_castle_ability[0]:
                    moves.append(encode_move(new_pos+1, new_pos-1, 5,
                                             EMPTY, 2, 0))
                elif pos == H1 and board[new_pos-1] == 5 and position.own_castle_ability[1]:
                    moves.append(encode_move(new_pos-1, new_pos+1, 5,
                                             EMPTY, 2, 0))

    return moves


@nb.njit
def get_pseudo_legal_captures(position):

    moves = []
    board = position.board
    for i in range(64):
        pos = STANDARD_TO_MAILBOX[i]
        piece = board[pos]
        if piece > 5:  # if it's not own piece
            continue
        for increment in OWN_ATK_INCREMENTS[piece]:
            if increment == 0:
                break
            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == PADDING or occupied < 6:  # if outside of board or own piece
                    break
                if occupied < EMPTY:
                    moves.append(encode_move(pos, new_pos,
                                             piece, occupied,
                                             0, 0))
                    break
                if piece in (0, 1, 5):  # if it is an opposing pawn, knight, or king
                    break
    return moves


@nb.njit
def get_scored_moves(engine, moves, tt_move):
    scored_moves = []

    for move in moves:
        scored_moves.append((move, score_move(engine, move, tt_move)))

    return scored_moves


@nb.njit
def get_scored_captures(moves):
    scored_moves = []

    for move in moves:
        scored_moves.append((move, score_capture(move)))

    return scored_moves


@nb.njit
def sort_next_move(moves, current_count):
    for next_count in range(current_count, len(moves)):
        if moves[current_count][1] < moves[next_count][1]:
            current_scored_move = moves[current_count]
            moves[current_count] = moves[next_count]
            moves[next_count] = current_scored_move
