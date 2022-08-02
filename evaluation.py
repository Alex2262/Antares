
from move import *
from utilities import *         # contains Evaluation arrays
from search_class import Search


# @nb.njit(SCORE_TYPE(Position.class_type.instance_type), cache=True)
@nb.njit
def evaluate(position):
    white_mid_scores = 0
    black_mid_scores = 0

    white_end_scores = 0
    black_end_scores = 0

    white_mid_piece_vals = 0
    black_mid_piece_vals = 0

    white_end_piece_vals = 0
    black_end_piece_vals = 0

    game_phase = 0
    board = position.board

    for i in range(64):
        piece = board[STANDARD_TO_MAILBOX[i]]
        if piece < BLACK_PAWN:
            white_mid_piece_vals += PIECE_VALUES[piece]
            white_end_piece_vals += ENDGAME_PIECE_VALUES[piece]
            white_mid_scores += PST[piece][i]
            white_end_scores += ENDGAME_PST[piece][i]

            game_phase += GAME_PHASE_SCORES[piece]

        elif piece < EMPTY:
            black_mid_piece_vals += PIECE_VALUES[piece - 6]
            black_end_piece_vals += ENDGAME_PIECE_VALUES[piece - 6]
            black_mid_scores += PST[piece - 6][i ^ 56]
            black_end_scores += ENDGAME_PST[piece - 6][i ^ 56]

            game_phase += GAME_PHASE_SCORES[piece-6]

    white_score = ((white_mid_scores + white_mid_piece_vals) * game_phase +
                   (24 - game_phase) * (white_end_scores + white_end_piece_vals)) / 24

    black_score = ((black_mid_scores + black_end_piece_vals) * game_phase +
                   (24 - game_phase) * (black_end_scores + black_end_piece_vals)) / 24

    return (position.side * -2 + 1) * SCORE_TYPE(white_score - black_score)


# @nb.njit(SCORE_TYPE(Search.class_type.instance_type, MOVE_TYPE, MOVE_TYPE), cache=True)
@nb.njit
def score_move(engine, move, tt_move):

    if move == tt_move:
        return 100000

    score = 0
    standard_from_square = MAILBOX_TO_STANDARD[get_from_square(move)]
    standard_to_square = MAILBOX_TO_STANDARD[get_to_square(move)]

    selected = get_selected(move)
    occupied = get_occupied(move)
    move_type = get_move_type(move)

    if selected < BLACK_PAWN:
        if get_is_capture(move):
            score += 10000
            score += 2 * (PIECE_VALUES[occupied - BLACK_PAWN] - PIECE_VALUES[selected])
            score += PST[occupied - BLACK_PAWN][standard_to_square ^ 56]
        else:
            # score 1st killer move
            if engine.killer_moves[0][engine.ply] == move:
                score += 9000
            # score 2nd killer move
            elif engine.killer_moves[1][engine.ply] == move:
                score += 8000
            # score history move
            else:
                score += engine.history_moves[selected][standard_to_square]

        if move_type == 3:  # Promotions
            score += 15000
            score += PIECE_VALUES[get_promotion_piece(move)]

        elif move_type == 2:  # Castling
            score += 1000

        elif move_type == 1:  # En Passant
            score += 2000

        score += PST[selected][standard_to_square] - \
                 PST[selected][standard_from_square]

    else:
        if get_is_capture(move):
            score += 10000
            score += 2 * (PIECE_VALUES[occupied] - PIECE_VALUES[selected - BLACK_PAWN])
            score += PST[occupied][standard_to_square]
        else:
            # score 1st killer move
            if engine.killer_moves[0][engine.ply] == move:
                score += 9000
            # score 2nd killer move
            elif engine.killer_moves[1][engine.ply] == move:
                score += 8000
            # score history move
            else:
                score += engine.history_moves[selected - BLACK_PAWN][standard_to_square]

        if move_type == 3:  # Promotions
            score += 15000
            score += PIECE_VALUES[get_promotion_piece(move) - BLACK_PAWN]

        elif move_type == 2:  # Castling
            score += 1000

        elif move_type == 1:  # En Passant
            score += 2000

        score += PST[selected - BLACK_PAWN][standard_to_square ^ 56] - \
                 PST[selected - BLACK_PAWN][standard_from_square ^ 56]

    return score


# @nb.njit(SCORE_TYPE(MOVE_TYPE), cache=True)
@nb.njit
def score_capture(move):

    score = 0

    standard_from_square = MAILBOX_TO_STANDARD[get_from_square(move)]
    standard_to_square = MAILBOX_TO_STANDARD[get_to_square(move)]

    selected = get_selected(move)
    occupied = get_occupied(move)

    if selected < BLACK_PAWN:
        score += 8 * (PIECE_VALUES[occupied - BLACK_PAWN] - PIECE_VALUES[selected])
        score += PST[occupied - BLACK_PAWN][standard_to_square]

        score += PST[selected][standard_to_square] - \
                 PST[selected][standard_from_square]
    else:
        score += 8 * (PIECE_VALUES[occupied] - PIECE_VALUES[selected - BLACK_PAWN])
        score += PST[occupied][standard_to_square ^ 56]

        score += PST[selected - BLACK_PAWN][standard_to_square ^ 56] - \
                 PST[selected - BLACK_PAWN][standard_from_square ^ 56]

    return score
