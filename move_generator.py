

from evaluation import score_move, score_capture
from move import *
# from position_class import Position
# from search_class import Search


# @nb.njit(nb.types.List(MOVE_TYPE)(Position.class_type.instance_type), cache=True)
@nb.njit
def get_pseudo_legal_moves(position):
    moves = []
    board = position.board

    if position.side == 0:  # white
        for pos in position.white_pieces:
            piece = board[pos]

            for increment in WHITE_INCREMENTS[piece]:
                if increment == 0:
                    break

                new_pos = pos
                while True:
                    new_pos += increment
                    occupied = board[new_pos]
                    if occupied == PADDING or occupied < BLACK_PAWN:  # standing on own piece or outside of board
                        break
                    elif piece == WHITE_PAWN and increment in (-10, -20) and board[pos-10] != EMPTY:
                        break
                    elif piece == WHITE_PAWN and increment == -20 and (pos < 81 or occupied != EMPTY):
                        break

                    # En passant
                    if piece == WHITE_PAWN and increment in (-11, -9) and occupied == EMPTY:
                        if new_pos == position.ep_square:
                            moves.append(encode_move(pos, new_pos,
                                                     WHITE_PAWN, EMPTY,
                                                     MOVE_TYPE_EP, 0, 0))
                        break

                    # Promotion
                    elif piece == WHITE_PAWN and new_pos < 31:
                        for j in range(WHITE_KNIGHT, WHITE_KING):
                            moves.append(encode_move(pos, new_pos,
                                                     WHITE_PAWN, occupied,
                                                     MOVE_TYPE_PROMOTION, j, 1 if occupied < EMPTY else 0))
                        break

                    # Normal capture move
                    if occupied < EMPTY:
                        moves.append(encode_move(pos, new_pos,
                                                 piece, occupied,
                                                 MOVE_TYPE_NORMAL, 0, 1))
                        break

                    # Normal non-capture move
                    moves.append(encode_move(pos, new_pos,
                                             piece, occupied,
                                             MOVE_TYPE_NORMAL, 0, 0))

                    # if we are a non-sliding piece, or we have captured an opposing piece then stop
                    if piece in (WHITE_PAWN, WHITE_KNIGHT, WHITE_KING):
                        break

                    # King side castle
                    if position.castle_ability_bits & 1 == 1 and pos == H1 and board[new_pos-1] == WHITE_KING:
                        moves.append(encode_move(E1, G1, WHITE_KING,
                                                 EMPTY, MOVE_TYPE_CASTLE, 0, 0))
                    # Queen side castle
                    elif position.castle_ability_bits & 2 == 2 and pos == A1 and board[new_pos+1] == WHITE_KING:
                        moves.append(encode_move(E1, C1, WHITE_KING,
                                                 EMPTY, MOVE_TYPE_CASTLE, 0, 0))

    else:
        for pos in position.black_pieces:
            piece = board[pos]

            for increment in BLACK_INCREMENTS[piece - BLACK_PAWN]:
                if increment == 0:
                    break

                new_pos = pos

                while True:
                    new_pos += increment
                    occupied = board[new_pos]
                    if occupied != EMPTY and occupied > WHITE_KING:  # standing on own piece or outside of board
                        break
                    elif piece == BLACK_PAWN and increment in (10, 20) and board[pos + 10] != EMPTY:
                        break
                    elif piece == BLACK_PAWN and increment == 20 and (pos > 38 or occupied != EMPTY):
                        break

                    # En passant
                    if piece == BLACK_PAWN and increment in (11, 9) and occupied == EMPTY:
                        if new_pos == position.ep_square:
                            moves.append(encode_move(pos, new_pos,
                                                     BLACK_PAWN, EMPTY,
                                                     MOVE_TYPE_EP, 0, 0))
                        break

                    # Promotion
                    elif piece == BLACK_PAWN and new_pos > 88:
                        for j in range(BLACK_KNIGHT, BLACK_KING):
                            moves.append(encode_move(pos, new_pos,
                                                     BLACK_PAWN, occupied,
                                                     MOVE_TYPE_PROMOTION, j, 1 if occupied < BLACK_PAWN else 0))
                        break

                    # Normal capture move
                    if occupied < BLACK_PAWN:
                        moves.append(encode_move(pos, new_pos,
                                                 piece, occupied,
                                                 MOVE_TYPE_NORMAL, 0, 1))
                        break
                    # Normal non-capture move
                    moves.append(encode_move(pos, new_pos,
                                             piece, occupied,
                                             MOVE_TYPE_NORMAL, 0, 0))

                    # if we are a non-sliding piece, or we have captured an opposing piece then stop
                    if piece in (BLACK_PAWN, BLACK_KNIGHT, BLACK_KING):
                        break

                    # King side castle
                    if position.castle_ability_bits & 4 == 4 and pos == H8 and board[new_pos-1] == BLACK_KING:
                        moves.append(encode_move(E8, G8, BLACK_KING,
                                                 EMPTY, MOVE_TYPE_CASTLE, 0, 0))
                    # Queen side castle
                    elif position.castle_ability_bits & 8 == 8 and pos == A8 and board[new_pos+1] == BLACK_KING:
                        moves.append(encode_move(E8, C8, BLACK_KING,
                                                 EMPTY, MOVE_TYPE_CASTLE, 0, 0))

    return moves


# @nb.njit(nb.types.List(MOVE_TYPE)(Position.class_type.instance_type), cache=True)
@nb.njit
def get_pseudo_legal_captures(position):

    moves = []
    board = position.board

    if position.side == 0:  # white
        for pos in position.white_pieces:
            piece = board[pos]

            for increment in WHITE_ATK_INCREMENTS[piece]:
                if increment == 0:
                    break

                new_pos = pos
                while True:
                    new_pos += increment
                    occupied = board[new_pos]

                    if occupied == PADDING or occupied < BLACK_PAWN:  # if outside of board or own piece
                        break
                    if occupied < EMPTY:
                        moves.append(encode_move(pos, new_pos,
                                                 piece, occupied,
                                                 MOVE_TYPE_NORMAL, 0, 1))
                        break
                    if piece in (WHITE_PAWN, WHITE_KNIGHT, WHITE_KING):  # if it is an opposing pawn, knight, or king
                        break
    else:
        for pos in position.black_pieces:
            piece = board[pos]

            for increment in BLACK_ATK_INCREMENTS[piece - BLACK_PAWN]:
                if increment == 0:
                    break

                new_pos = pos
                while True:
                    new_pos += increment
                    occupied = board[new_pos]

                    if occupied != EMPTY and occupied > WHITE_KING:  # standing on own piece or outside of board
                        break

                    if occupied < BLACK_PAWN:
                        moves.append(encode_move(pos, new_pos,
                                                 piece, occupied,
                                                 MOVE_TYPE_NORMAL, 0, 1))
                        break
                    if piece in (BLACK_PAWN, BLACK_KNIGHT, BLACK_KING):  # if it is an opposing pawn, knight, or king
                        break

    return moves


# @nb.njit(nb.types.List(SCORE_TYPE)(Search.class_type.instance_type, nb.types.List(MOVE_TYPE), MOVE_TYPE))
@nb.njit
def get_move_scores(engine, moves, tt_move):
    scored_moves = []

    for move in moves:
        scored_moves.append(score_move(engine, move, tt_move))

    return scored_moves


# @nb.njit(nb.types.List(SCORE_TYPE)(nb.types.List(MOVE_TYPE)), cache=True)
@nb.njit
def get_capture_scores(moves):
    scored_moves = []

    for move in moves:
        scored_moves.append(score_capture(move))

    return scored_moves


# @nb.njit(nb.void(nb.types.List(MOVE_TYPE), nb.types.List(SCORE_TYPE), nb.uint8), cache=True)
@nb.njit
def sort_next_move(moves, move_scores, current_count):
    for next_count in range(current_count, len(moves)):
        if move_scores[current_count] < move_scores[next_count]:
            current_move = moves[current_count]
            moves[current_count] = moves[next_count]
            moves[next_count] = current_move

            current_score = move_scores[current_count]
            move_scores[current_count] = move_scores[next_count]
            move_scores[next_count] = current_score
