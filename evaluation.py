
from move import *
from utilities import *         # contains Evaluation arrays
from search_class import Search


@nb.njit
def evaluate_pawn(position, pawn_rank, pos):
    i = MAILBOX_TO_STANDARD[pos]
    f = i % 8 + 1  # pawn's file ( Col + 1 )
    row = 8 - i // 8

    mid_score = 0
    end_score = 0
    side = 0 if position.board[pos] == WHITE_PAWN else 1

    if side == 0:

        if pawn_rank[0][f] > row:
            mid_score -= DOUBLED_PAWN_PENALTY
            end_score -= ENDGAME_DOUBLED_PAWN_PENALTY

        '''if pawn_rank[0][f - 1] == 0 and pawn_rank[0][f + 1] == 0:
            
            if pawn_rank[1][f] == 9:
                # The isolated pawn in the middle game is worse if the opponent
                # has the semi open file to attack it.
                mid_score -= 1.5 * ISOLATED_PAWN_PENALTY
                
                # In the endgame it can be slightly better since it has the chance to become passed
                end_score -= 0.8 * ENDGAME_ISOLATED_PAWN_PENALTY
            else:
                mid_score -= ISOLATED_PAWN_PENALTY
                end_score -= ENDGAME_ISOLATED_PAWN_PENALTY

        elif pawn_rank[0][f - 1] > row and pawn_rank[0][f + 1] > row:
            # In the middle game it's worse to have a very backwards pawn
            # since then, the 'forwards' pawns won't be protected
            mid_score -= BACKWARDS_PAWN_PENALTY + \
                         2 * (pawn_rank[0][f - 1] - row + pawn_rank[0][f + 1] - row - 2)

            # In the end game the backwards pawn should be worse, but if it's very backwards it's not awful.
            end_score -= ENDGAME_BACKWARDS_PAWN_PENALTY + \
                     pawn_rank[0][f - 1] - row + pawn_rank[0][f + 1] - row - 2

            # If there's no enemy pawn in front of our pawn then it's even worse, since
            # we allow outposts and pieces to attack us easily
            if pawn_rank[1][f] == 9:
                # In the middle game it is worse since enemy pieces can use the semi-open file and outpost.
                mid_score -= 3 * BACKWARDS_PAWN_PENALTY
                end_score -= BACKWARDS_PAWN_PENALTY

        if pawn_rank[1][f - 1] <= row and\
             pawn_rank[1][f] <= row and\
             pawn_rank[1][f + 1] <= row:
            mid_score += row * PASSED_PAWN_BONUS
            end_score += row * ENDGAME_PASSED_PAWN_BONUS'''

    else:

        if pawn_rank[1][f] < row:
            mid_score -= DOUBLED_PAWN_PENALTY
            end_score -= ENDGAME_DOUBLED_PAWN_PENALTY

    return mid_score, end_score


# @nb.njit(SCORE_TYPE(Position.class_type.instance_type), cache=True)
@nb.njit
def evaluate(position):
    white_mid_scores = 0
    black_mid_scores = 0

    white_end_scores = 0
    black_end_scores = 0

    pawn_rank = np.zeros((2, 10), dtype=nb.uint8)

    game_phase = 0
    board = position.board

    for i in range(10):
        # This can tell us if there's no pawn in this file
        pawn_rank[0][i] = 0
        pawn_rank[1][i] = 9

    for i in range(64):
        pos = STANDARD_TO_MAILBOX[i]
        piece = board[pos]
        if piece == WHITE_PAWN:
            if pawn_rank[0][i + 1] < 8 - i // 8:
                pawn_rank[0][i + 1] = 8 - i // 8

        elif piece == BLACK_PAWN:
            if pawn_rank[1][i + 1] > 8 - i // 8:
                pawn_rank[1][i + 1] = 8 - i // 8

    for i in range(64):
        pos = STANDARD_TO_MAILBOX[i]
        piece = board[pos]
        if piece < BLACK_PAWN:
            white_mid_scores += PIECE_VALUES[piece]
            white_end_scores += ENDGAME_PIECE_VALUES[piece]
            white_mid_scores += PST[piece][i]
            white_end_scores += ENDGAME_PST[piece][i]

            game_phase += GAME_PHASE_SCORES[piece]

            if piece == WHITE_PAWN:
                pawn_scores = evaluate_pawn(position, pawn_rank, pos)
                white_mid_scores += pawn_scores[0]
                white_end_scores += pawn_scores[1]

        elif piece < EMPTY:
            black_mid_scores += PIECE_VALUES[piece - 6]
            black_end_scores += ENDGAME_PIECE_VALUES[piece - 6]
            black_mid_scores += PST[piece - 6][i ^ 56]
            black_end_scores += ENDGAME_PST[piece - 6][i ^ 56]

            game_phase += GAME_PHASE_SCORES[piece-6]

            if piece == BLACK_PAWN:
                pawn_scores = evaluate_pawn(position, pawn_rank, pos)
                black_mid_scores += pawn_scores[0]
                black_end_scores += pawn_scores[1]

    white_score = (white_mid_scores * game_phase +
                   (24 - game_phase) * white_end_scores) / 24

    black_score = (black_mid_scores * game_phase +
                   (24 - game_phase) * black_end_scores) / 24

    return (position.side * -2 + 1) * SCORE_TYPE(white_score - black_score + TEMPO_BONUS)


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
