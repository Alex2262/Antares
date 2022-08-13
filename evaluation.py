
from move import *
from utilities import *         # contains Evaluation arrays
from search_class import Search


@nb.njit(cache=True)
def evaluate_pawn(position, pawn_rank, pos):
    i = MAILBOX_TO_STANDARD[pos]
    f = i % 8 + 1  # pawn's file ( Col + 1 )
    row = 8 - i // 8

    mid_score = 0
    end_score = 0

    pawn_color = 0 if position.board[pos] == WHITE_PAWN else 1
    if pawn_color == 0:

        # Doubled pawns. The pawn we are checking is higher in row compared to
        # the least advanced pawn in our column.
        if row > pawn_rank[0][f]:
            mid_score -= DOUBLED_PAWN_PENALTY
            end_score -= ENDGAME_DOUBLED_PAWN_PENALTY

        # Isolated pawns. We do not have pawns on the columns next to our pawn.
        if pawn_rank[0][f - 1] == 9 and pawn_rank[0][f + 1] == 9:

            # If our opponent does not have a pawn in front of our pawn
            if pawn_rank[1][f] == 0:
                # The isolated pawn in the middle game is worse if the opponent
                # has the semi open file to attack it.
                mid_score -= 1.5 * ISOLATED_PAWN_PENALTY
                
                # In the endgame it can be slightly better since it has the chance to become passed
                end_score -= 0.8 * ENDGAME_ISOLATED_PAWN_PENALTY
            else:
                mid_score -= ISOLATED_PAWN_PENALTY
                end_score -= ENDGAME_ISOLATED_PAWN_PENALTY

        elif row < pawn_rank[0][f - 1] and row < pawn_rank[0][f + 1]:
            # In the middle game it's worse to have a very backwards pawn
            # since then, the 'forwards' pawns won't be protected
            mid_score -= BACKWARDS_PAWN_PENALTY + \
                         2 * (pawn_rank[0][f - 1] - row + pawn_rank[0][f + 1] - row - 2)

            # In the end game the backwards pawn should be worse, but if it's very backwards it's not awful.
            end_score -= ENDGAME_BACKWARDS_PAWN_PENALTY + \
                     pawn_rank[0][f - 1] - row + pawn_rank[0][f + 1] - row - 2

            # If there's no enemy pawn in front of our pawn then it's even worse, since
            # we allow outposts and pieces to attack us easily
            if pawn_rank[1][f] == 0:
                # In the endgame with no pieces it wouldn't be a big deal, in some situations it could be better.
                mid_score -= 3 * BACKWARDS_PAWN_PENALTY

        if row >= pawn_rank[1][f - 1] and\
                row >= pawn_rank[1][f] and\
                row >= pawn_rank[1][f + 1]:
            mid_score += row * PASSED_PAWN_BONUS
            end_score += row * ENDGAME_PASSED_PAWN_BONUS

    else:

        if row < pawn_rank[1][f]:
            mid_score -= DOUBLED_PAWN_PENALTY
            end_score -= ENDGAME_DOUBLED_PAWN_PENALTY

        if pawn_rank[1][f - 1] == 0 and pawn_rank[1][f + 1] == 0:

            if pawn_rank[0][f] == 9:
                # The isolated pawn in the middle game is worse if the opponent
                # has the semi open file to attack it.
                mid_score -= 1.5 * ISOLATED_PAWN_PENALTY

                # In the endgame it can be slightly better since it has the chance to become passed
                end_score -= 0.8 * ENDGAME_ISOLATED_PAWN_PENALTY
            else:
                mid_score -= ISOLATED_PAWN_PENALTY
                end_score -= ENDGAME_ISOLATED_PAWN_PENALTY

        elif row < pawn_rank[1][f - 1] and row < pawn_rank[1][f + 1]:
            # In the middle game it's worse to have a very backwards pawn
            # since then, the 'forwards' pawns won't be protected
            mid_score -= BACKWARDS_PAWN_PENALTY + \
                         2 * (row - pawn_rank[1][f - 1] + row - pawn_rank[1][f + 1] - 2)

            # In the end game the backwards pawn should be worse, but if it's very backwards it's not awful.
            end_score -= ENDGAME_BACKWARDS_PAWN_PENALTY + \
                         row - pawn_rank[1][f - 1] + row - pawn_rank[1][f + 1] - 2

            # If there's no enemy pawn in front of our pawn then it's even worse, since
            # we allow outposts and pieces to attack us easily
            if pawn_rank[0][f] == 0:
                # In the middle game it is worse since enemy pieces can use the semi-open file and outpost.
                mid_score -= 3 * BACKWARDS_PAWN_PENALTY

        if row <= pawn_rank[0][f - 1] and \
                row <= pawn_rank[0][f] and \
                row <= pawn_rank[0][f + 1]:
            mid_score += (9 - row) * PASSED_PAWN_BONUS
            end_score += (9 - row) * ENDGAME_PASSED_PAWN_BONUS

    return mid_score, end_score


@nb.njit(cache=True)
def evaluate_king_pawn(pawn_rank, file, color):

    score = 0

    if color == 0:
        if pawn_rank[0][file] == 3:  # Pawn moved one square
            score -= 6
        elif pawn_rank[0][file] == 4:  # Pawn moved two squares
            score -= 20
        elif pawn_rank[0][file] != 2:  # Pawn moved more than two squares, or no pawn on this file
            score -= 27

        if pawn_rank[1][file] == 0:  # no enemy pawn on this file
            score -= 18
        elif pawn_rank[1][file] == 4:  # Enemy pawn is on the 4th rank
            score -= 8
        elif pawn_rank[1][file] == 3:  # Enemy pawn is on the 3rd rank
            score -= 15

    else:
        if pawn_rank[1][file] == 6:
            score -= 6
        elif pawn_rank[1][file] == 5:
            score -= 20
        elif pawn_rank[1][file] != 7:
            score -= 27

        if pawn_rank[0][file] == 9:
            score -= 18
        elif pawn_rank[0][file] == 5:
            score -= 8
        elif pawn_rank[0][file] == 6:
            score -= 15

    return score


@nb.njit(cache=True)
def evaluate_king(position, pawn_rank, pos):
    i = MAILBOX_TO_STANDARD[pos]
    col = i % 8

    # Only a middle game score is needed, since king safety isn't of concern in the endgame.
    score = 0

    king_color = 0 if position.board[pos] == WHITE_KING else 1

    if king_color == 0:
        if col < 3:  # Queen side
            score += evaluate_king_pawn(pawn_rank, 1, 0) * 0.8  # A file pawn
            score += evaluate_king_pawn(pawn_rank, 2, 0)
            score += evaluate_king_pawn(pawn_rank, 3, 0) * 0.6  # C file pawn

        elif col > 4:
            score += evaluate_king_pawn(pawn_rank, 8, 0) * 0.5  # H file pawn
            score += evaluate_king_pawn(pawn_rank, 7, 0)
            score += evaluate_king_pawn(pawn_rank, 6, 0) * 0.3  # F file pawn

        else:
            for pawn_file in range(col, col + 3):
                if pawn_rank[0][pawn_file] == 9:
                    score -= 7
                    if pawn_rank[1][pawn_file] == 0:
                        score -= 15

    else:
        if col < 3:  # Queen side
            score += evaluate_king_pawn(pawn_rank, 1, 1) * 0.8  # A file pawn
            score += evaluate_king_pawn(pawn_rank, 2, 1)
            score += evaluate_king_pawn(pawn_rank, 3, 1) * 0.6  # C file pawn

        elif col > 4:
            score += evaluate_king_pawn(pawn_rank, 8, 1) * 0.5  # H file pawn
            score += evaluate_king_pawn(pawn_rank, 7, 1)
            score += evaluate_king_pawn(pawn_rank, 6, 1) * 0.3  # F file pawn

        else:
            for pawn_file in range(col, col + 3):
                if pawn_rank[1][pawn_file] == 0:
                    score -= 7
                    if pawn_rank[0][pawn_file] == 9:
                        score -= 15

    return score


# @nb.njit(SCORE_TYPE(Position.class_type.instance_type), cache=True)
@nb.njit(cache=True)
def evaluate(position):
    white_mid_scores = 0
    black_mid_scores = 0

    white_end_scores = 0
    black_end_scores = 0

    # We make a 10 size array for each side, and eight of them are used for storing
    # the least advanced pawn. Storing this allows us to check for passed pawns,
    # backwards pawns, isolated pawns and whatnot.
    # Having a ten element array gives padding on the side to prevent out of bounds exceptions.
    pawn_rank = np.zeros((2, 10), dtype=np.uint8)

    white_bishops = 0
    black_bishops = 0

    game_phase = 0
    board = position.board

    for i in range(10):
        # This can tell us if there's no pawn in this file
        pawn_rank[0][i] = 9
        pawn_rank[1][i] = 0

    for pos in position.white_pieces:
        piece = board[pos]
        i = MAILBOX_TO_STANDARD[pos]
        row = 8 - i // 8
        f = i % 8 + 1
        if piece == WHITE_PAWN:
            if row < pawn_rank[0][f]:
                pawn_rank[0][f] = row

    for pos in position.black_pieces:
        piece = board[pos]
        i = MAILBOX_TO_STANDARD[pos]
        row = 8 - i // 8
        f = i % 8 + 1

        if piece == BLACK_PAWN:
            if row > pawn_rank[1][f]:
                pawn_rank[1][f] = row

    for pos in position.white_pieces:

        piece = board[pos]
        i = MAILBOX_TO_STANDARD[pos]

        white_mid_scores += PIECE_VALUES[piece]
        white_end_scores += ENDGAME_PIECE_VALUES[piece]
        white_mid_scores += PST[piece][i]
        white_end_scores += ENDGAME_PST[piece][i]

        game_phase += GAME_PHASE_SCORES[piece]

        if piece == WHITE_PAWN:
            pawn_scores = evaluate_pawn(position, pawn_rank, pos)
            white_mid_scores += pawn_scores[0]
            white_end_scores += pawn_scores[1]

        elif piece == WHITE_BISHOP:
            white_bishops += 1

        elif piece == WHITE_ROOK:
            if pawn_rank[0][i % 8 + 1] == 9:  # No pawn on this column
                if pawn_rank[1][i % 8 + 1] == 0:  # No enemy pawn on column
                    white_mid_scores += ROOK_OPEN_FILE_BONUS
                    white_end_scores += ENDGAME_ROOK_OPEN_FILE_BONUS
                else:
                    white_mid_scores += ROOK_SEMI_OPEN_FILE_BONUS
                    white_end_scores += ENDGAME_ROOK_SEMI_OPEN_FILE_BONUS

        elif piece == WHITE_KING:
            white_mid_scores += evaluate_king(position, pawn_rank, pos)

    for pos in position.black_pieces:

        piece = board[pos]
        i = MAILBOX_TO_STANDARD[pos]

        black_mid_scores += PIECE_VALUES[piece - 6]
        black_end_scores += ENDGAME_PIECE_VALUES[piece - 6]
        black_mid_scores += PST[piece - 6][i ^ 56]
        black_end_scores += ENDGAME_PST[piece - 6][i ^ 56]

        game_phase += GAME_PHASE_SCORES[piece-6]

        if piece == BLACK_PAWN:
            pawn_scores = evaluate_pawn(position, pawn_rank, pos)
            black_mid_scores += pawn_scores[0]
            black_end_scores += pawn_scores[1]

        elif piece == BLACK_BISHOP:
            black_bishops += 1

        elif piece == BLACK_ROOK:
            if pawn_rank[1][i % 8 + 1] == 0:  # No pawn on this column
                if pawn_rank[0][i % 8 + 1] == 9:  # No enemy pawn on column
                    black_mid_scores += ROOK_OPEN_FILE_BONUS
                    black_end_scores += ENDGAME_ROOK_OPEN_FILE_BONUS
                else:
                    black_mid_scores += ROOK_SEMI_OPEN_FILE_BONUS
                    black_end_scores += ENDGAME_ROOK_SEMI_OPEN_FILE_BONUS

        elif piece == BLACK_KING:
            black_mid_scores += evaluate_king(position, pawn_rank, pos)

    if white_bishops >= 2:
        white_mid_scores += BISHOP_PAIR_BONUS
        white_end_scores += ENDGAME_BISHOP_PAIR_BONUS

    if black_bishops >= 2:
        black_mid_scores += BISHOP_PAIR_BONUS
        black_end_scores += ENDGAME_BISHOP_PAIR_BONUS

    game_phase = min(game_phase, 24)  # promotions
    white_score = (white_mid_scores * game_phase +
                   (24 - game_phase) * white_end_scores) / 24

    black_score = (black_mid_scores * game_phase +
                   (24 - game_phase) * black_end_scores) / 24

    return SCORE_TYPE((position.side * -2 + 1) * (white_score - black_score) + TEMPO_BONUS)


# @nb.njit(SCORE_TYPE(Search.class_type.instance_type, MOVE_TYPE, MOVE_TYPE), cache=True)
@nb.njit(cache=True)
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
                score += 500 + engine.history_moves[selected][standard_to_square]

        if move_type == 3:  # Promotions
            score += 15000 + PIECE_VALUES[get_promotion_piece(move)]

        elif move_type == 2:  # Castling
            score += 1000     # Helps to incentivize castling

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
                score += 500 + engine.history_moves[selected][standard_to_square]

        if move_type == 3:  # Promotions
            score += 15000 + PIECE_VALUES[get_promotion_piece(move) - BLACK_PAWN]

        elif move_type == 2:  # Castling
            score += 1000

        elif move_type == 1:  # En Passant
            score += 2000

        score += PST[selected - BLACK_PAWN][standard_to_square ^ 56] - \
                 PST[selected - BLACK_PAWN][standard_from_square ^ 56]

    return score


# @nb.njit(SCORE_TYPE(MOVE_TYPE), cache=True)
@nb.njit(cache=True)
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
