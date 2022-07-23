
from move import *
from utilities import *         # contains Evaluation arrays
#from search import Search


@nb.njit
def evaluate(position):
    own_mid_scores = 0
    opp_mid_scores = 0

    own_end_scores = 0
    opp_end_scores = 0

    own_mid_piece_vals = 0
    opp_mid_piece_vals = 0

    own_end_piece_vals = 0
    opp_end_piece_vals = 0

    board = position.board

    for i in range(64):
        piece = board[STANDARD_TO_MAILBOX[i]]
        if piece < 6:
            own_mid_piece_vals += PIECE_VALUES[piece]
            own_end_piece_vals += ENDGAME_PIECE_VALUES[piece]
            own_mid_scores += PST[piece][i]
            own_end_scores += ENDGAME_PST[piece][i]

        elif piece < 12:
            opp_mid_piece_vals += PIECE_VALUES[piece - 6]
            opp_end_piece_vals += ENDGAME_PIECE_VALUES[piece - 6]
            opp_mid_scores += PST[piece - 6][i ^ 56]
            opp_end_scores += ENDGAME_PST[piece - 6][i ^ 56]

    if own_end_piece_vals < 1300:
        opp_score = opp_end_scores + opp_end_piece_vals
    else:
        opp_score = opp_mid_scores + opp_mid_piece_vals

    if opp_end_piece_vals < 1300:
        own_score = own_end_scores + own_end_piece_vals
    else:
        own_score = own_mid_scores + own_mid_piece_vals

    return own_score-opp_score


@nb.njit(cache=False)
def score_move(engine, move, tt_move):

    if engine.score_pv:
        if engine.pv_table[0][engine.ply] == move:
            engine.score_pv = False
            return 30000

    if move == tt_move:
        return 30000

    score = 0
    standard_from_square = MAILBOX_TO_STANDARD[get_from_square(move)]
    standard_to_square = MAILBOX_TO_STANDARD[get_to_square(move)]

    selected = get_selected(move)
    occupied = get_occupied(move)
    move_type = get_move_type(move)

    if get_is_capture(move):
        score += 10000
        score += PIECE_VALUES[occupied - 6] - PIECE_VALUES[selected]
        score += PST[occupied - 6][standard_to_square ^ 56]
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

    score += PST[selected][standard_to_square] - PST[selected][standard_from_square]

    return score


@nb.njit(cache=False)
def score_capture(move):

    score = 0

    standard_from_square = MAILBOX_TO_STANDARD[get_from_square(move)]
    standard_to_square = MAILBOX_TO_STANDARD[get_to_square(move)]

    selected = get_selected(move)
    occupied = get_occupied(move)

    score += 8 * (PIECE_VALUES[occupied - 6] - PIECE_VALUES[selected])
    score += PST[occupied - 6][standard_to_square ^ 56]

    score += PST[selected][standard_to_square] - PST[selected][standard_from_square]

    return score
