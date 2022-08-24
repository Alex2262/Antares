

from move import *
from utilities import *         # contains Evaluation arrays


@nb.njit(cache=True)
def get_distance(square_one, square_two):
    row_one = 8 - square_one // 8
    row_two = 8 - square_two // 8

    col_one = square_one % 8 + 1
    col_two = square_two % 8 + 1

    return (abs(row_one - row_two) ** 2 + abs(col_one - col_two) ** 2) ** 0.5


@nb.njit(cache=True)
def evaluate_pawn(position, pos, pawn_rank):
    i = MAILBOX_TO_STANDARD[pos]
    col = i % 8 + 1  # pawn's file ( Col + 1 )
    row = 8 - i // 8

    mid_score = 0
    end_score = 0

    piece_color = 0 if position.board[pos] == WHITE_PAWN else 1
    if piece_color == 0:

        mid_score += PAWN_PST_MID[i]
        end_score += PAWN_PST_END[i]

        # Doubled pawns. The pawn we are checking is higher in row compared to
        # the least advanced pawn in our column.
        if row > pawn_rank[0][col]:
            mid_score -= DOUBLED_PAWN_PENALTY_MID
            end_score -= DOUBLED_PAWN_PENALTY_END

        # Isolated pawns. We do not have pawns on the columns next to our pawn.
        if pawn_rank[0][col - 1] == 9 and pawn_rank[0][col + 1] == 9:

            # If our opponent does not have a pawn in front of our pawn
            if pawn_rank[1][col] == 0:
                # The isolated pawn in the middle game is worse if the opponent
                # has the semi open file to attack it.
                mid_score -= 1.5 * ISOLATED_PAWN_PENALTY_MID
                
                # In the endgame it can be slightly better since it has the chance to become passed
                end_score -= 0.8 * ISOLATED_PAWN_PENALTY_END
            else:
                mid_score -= ISOLATED_PAWN_PENALTY_MID
                end_score -= ISOLATED_PAWN_PENALTY_END

        elif row < pawn_rank[0][col - 1] and row < pawn_rank[0][col + 1]:
            # In the middle game it's worse to have a very backwards pawn
            # since then, the 'forwards' pawns won't be protected
            mid_score -= BACKWARDS_PAWN_PENALTY_MID + \
                         2 * (pawn_rank[0][col - 1] - row + pawn_rank[0][col + 1] - row - 2)

            # In the end game the backwards pawn should be worse, but if it's very backwards it's not awful.
            end_score -= BACKWARDS_PAWN_PENALTY_END + \
                     pawn_rank[0][col - 1] - row + pawn_rank[0][col + 1] - row - 2

            # If there's no enemy pawn in front of our pawn then it's even worse, since
            # we allow outposts and pieces to attack us easily
            if pawn_rank[1][col] == 0:
                # In the endgame with no pieces it wouldn't be a big deal, in some situations it could be better.
                mid_score -= 3 * BACKWARDS_PAWN_PENALTY_MID

        if row >= pawn_rank[1][col - 1]    \
             and row >= pawn_rank[1][col]  \
             and row >= pawn_rank[1][col + 1]:

            mid_score += row * PASSED_PAWN_BONUS_MID
            end_score += row * PASSED_PAWN_BONUS_END

    else:

        mid_score += PAWN_PST_MID[i ^ 56]
        end_score += PAWN_PST_END[i ^ 56]

        if row < pawn_rank[1][col]:
            mid_score -= DOUBLED_PAWN_PENALTY_MID
            end_score -= DOUBLED_PAWN_PENALTY_END

        if pawn_rank[1][col - 1] == 0 and pawn_rank[1][col + 1] == 0:

            if pawn_rank[0][col] == 9:
                # The isolated pawn in the middle game is worse if the opponent
                # has the semi open file to attack it.
                mid_score -= 1.5 * ISOLATED_PAWN_PENALTY_MID

                # In the endgame it can be slightly better since it has the chance to become passed
                end_score -= 0.8 * ISOLATED_PAWN_PENALTY_END
            else:
                mid_score -= ISOLATED_PAWN_PENALTY_MID
                end_score -= ISOLATED_PAWN_PENALTY_END

        elif row < pawn_rank[1][col - 1] and row < pawn_rank[1][col + 1]:
            # In the middle game it's worse to have a very backwards pawn
            # since then, the 'forwards' pawns won't be protected
            mid_score -= BACKWARDS_PAWN_PENALTY_MID + \
                         2 * (row - pawn_rank[1][col - 1] + row - pawn_rank[1][col + 1] - 2)

            # In the end game the backwards pawn should be worse, but if it's very backwards it's not awful.
            end_score -= BACKWARDS_PAWN_PENALTY_END + \
                         row - pawn_rank[1][col - 1] + row - pawn_rank[1][col + 1] - 2

            # If there's no enemy pawn in front of our pawn then it's even worse, since
            # we allow outposts and pieces to attack us easily
            if pawn_rank[0][col] == 0:
                # In the middle game it is worse since enemy pieces can use the semi-open file and outpost.
                mid_score -= 3 * BACKWARDS_PAWN_PENALTY_MID

        if row <= pawn_rank[0][col - 1]   \
             and row <= pawn_rank[0][col] \
             and row <= pawn_rank[0][col + 1]:

            mid_score += (9 - row) * PASSED_PAWN_BONUS_MID
            end_score += (9 - row) * PASSED_PAWN_BONUS_END

    return mid_score, end_score


@nb.njit(cache=True)
def evaluate_king_pawn(piece_color, file, pawn_rank):

    score = 0

    if piece_color == 0:
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
def evaluate_knight(position, pos):

    piece = position.board[pos]
    i = MAILBOX_TO_STANDARD[pos]

    piece_color = 0 if piece == WHITE_KNIGHT else 1

    mid_score = 0
    end_score = 0

    if piece_color == 0:
        mid_score += KNIGHT_PST_MID[i]
        end_score += KNIGHT_PST_END[i]
    else:
        mid_score += KNIGHT_PST_MID[i ^ 56]
        end_score += KNIGHT_PST_END[i ^ 56]

    # We want our knights to be somewhat close to our king, for the purpose of king safety.
    distance_to_our_king = get_distance(i, MAILBOX_TO_STANDARD[position.king_positions[piece_color]])
    mid_score -= distance_to_our_king
    end_score -= 0.5 * distance_to_our_king

    # The closer our knights are to the opponent's king, the better we can attack.
    distance_to_opp_king = get_distance(i, MAILBOX_TO_STANDARD[position.king_positions[piece_color ^ 1]])
    mid_score -= 3.5 * distance_to_opp_king
    end_score -= 1.5 * distance_to_opp_king

    return mid_score, end_score


@nb.njit(cache=True)
def evaluate_bishop(position, pos):

    piece = position.board[pos]
    i = MAILBOX_TO_STANDARD[pos]

    piece_color = 0 if piece == WHITE_BISHOP else 1

    mid_score = 0
    end_score = 0

    if piece_color == 0:
        mid_score += BISHOP_PST_MID[i]
        end_score += BISHOP_PST_END[i]
    else:
        mid_score += BISHOP_PST_MID[i ^ 56]
        end_score += BISHOP_PST_END[i ^ 56]

    distance_to_opp_king = get_distance(i, MAILBOX_TO_STANDARD[position.king_positions[piece_color ^ 1]])
    mid_score -= 1.5 * distance_to_opp_king
    end_score -= 0.5 * distance_to_opp_king

    return mid_score, end_score


@nb.njit(cache=True)
def evaluate_rook(position, pos, pawn_rank):
    piece = position.board[pos]
    i = MAILBOX_TO_STANDARD[pos]
    col = i % 8 + 1

    piece_color = 0 if piece == WHITE_ROOK else 1

    mid_score = 0
    end_score = 0

    if piece_color == 0:
        mid_score += ROOK_PST_MID[i]
        end_score += ROOK_PST_END[i]

        if pawn_rank[0][col] == 9:  # No pawn on this column
            if pawn_rank[1][col] == 0:  # No enemy pawn on column
                mid_score += ROOK_OPEN_FILE_BONUS_MID
                end_score += ROOK_OPEN_FILE_BONUS_END
            else:
                mid_score += ROOK_SEMI_OPEN_FILE_BONUS_MID
                end_score += ROOK_SEMI_OPEN_FILE_BONUS_END

    else:
        mid_score += ROOK_PST_MID[i ^ 56]
        end_score += ROOK_PST_END[i ^ 56]

        if pawn_rank[1][col] == 0:  # No pawn on this column
            if pawn_rank[0][col] == 9:  # No enemy pawn on column
                mid_score += ROOK_OPEN_FILE_BONUS_MID
                end_score += ROOK_OPEN_FILE_BONUS_END
            else:
                mid_score += ROOK_SEMI_OPEN_FILE_BONUS_MID
                end_score += ROOK_SEMI_OPEN_FILE_BONUS_END

    distance_to_opp_king = get_distance(i, MAILBOX_TO_STANDARD[position.king_positions[piece_color ^ 1]])
    mid_score -= 3 * distance_to_opp_king
    end_score -= 1.5 * distance_to_opp_king
    return mid_score, end_score


@nb.njit(cache=True)
def evaluate_queen(position, pos, pawn_rank):
    piece = position.board[pos]
    i = MAILBOX_TO_STANDARD[pos]
    col = i % 8 + 1

    piece_color = 0 if piece == WHITE_QUEEN else 1

    mid_score = 0
    end_score = 0

    if piece_color == 0:
        mid_score += QUEEN_PST_MID[i]
        end_score += QUEEN_PST_END[i]

        if pawn_rank[0][col] == 9:  # No pawn on this column
            if pawn_rank[1][col] == 0:  # No enemy pawn on column
                mid_score += QUEEN_OPEN_FILE_BONUS_MID
                end_score += QUEEN_OPEN_FILE_BONUS_END
            else:
                mid_score += QUEEN_SEMI_OPEN_FILE_BONUS_MID
                end_score += QUEEN_SEMI_OPEN_FILE_BONUS_END

    else:
        mid_score += QUEEN_PST_MID[i ^ 56]
        end_score += QUEEN_PST_END[i ^ 56]

        if pawn_rank[1][col] == 0:  # No pawn on this column
            if pawn_rank[0][col] == 9:  # No enemy pawn on column
                mid_score += QUEEN_OPEN_FILE_BONUS_MID
                end_score += QUEEN_OPEN_FILE_BONUS_END
            else:
                mid_score += QUEEN_SEMI_OPEN_FILE_BONUS_MID
                end_score += QUEEN_SEMI_OPEN_FILE_BONUS_END

    distance_to_opp_king = get_distance(i, MAILBOX_TO_STANDARD[position.king_positions[piece_color ^ 1]])
    mid_score -= 1.5 * distance_to_opp_king
    end_score -= 2 * distance_to_opp_king

    return mid_score, end_score


@nb.njit(cache=True)
def evaluate_king(position, pos, pawn_rank):
    i = MAILBOX_TO_STANDARD[pos]
    col = i % 8 + 1

    mid_score = 0
    end_score = 0

    piece_color = 0 if position.board[pos] == WHITE_KING else 1

    if piece_color == 0:

        mid_score += KING_PST_MID[i]
        end_score += KING_PST_END[i]

        if col < 4:  # Queen side
            mid_score += evaluate_king_pawn(0, 1, pawn_rank) * 0.8  # A file pawn
            mid_score += evaluate_king_pawn(0, 2, pawn_rank)
            mid_score += evaluate_king_pawn(0, 3, pawn_rank) * 0.6  # C file pawn

        elif col > 5:
            mid_score += evaluate_king_pawn(0, 8, pawn_rank) * 0.5  # H file pawn
            mid_score += evaluate_king_pawn(0, 7, pawn_rank)
            mid_score += evaluate_king_pawn(0, 6, pawn_rank) * 0.3  # F file pawn

        else:
            for pawn_file in range(col - 1, col + 2):
                if pawn_rank[0][pawn_file] == 9:
                    mid_score -= 7
                    if pawn_rank[1][pawn_file] == 0:
                        mid_score -= 15

    else:

        mid_score += KING_PST_MID[i ^ 56]
        end_score += KING_PST_END[i ^ 56]

        if col < 4:  # Queen side
            mid_score += evaluate_king_pawn(1, 1, pawn_rank) * 0.8  # A file pawn
            mid_score += evaluate_king_pawn(1, 2, pawn_rank)
            mid_score += evaluate_king_pawn(1, 3, pawn_rank) * 0.6  # C file pawn

        elif col > 5:
            mid_score += evaluate_king_pawn(1, 8, pawn_rank) * 0.5  # H file pawn
            mid_score += evaluate_king_pawn(1, 7, pawn_rank)
            mid_score += evaluate_king_pawn(1, 6, pawn_rank) * 0.3  # F file pawn

        else:
            for pawn_file in range(col - 1, col + 2):
                if pawn_rank[1][pawn_file] == 0:
                    mid_score -= 7
                    if pawn_rank[0][pawn_file] == 9:
                        mid_score -= 15

    return mid_score, end_score


@nb.njit(cache=True)
def check_material_draw(white_piece_amounts, black_piece_amounts, white_material, black_material):

    # 2 == there's still chance for winning
    # 1 == close to a draw
    # 0 == dead draw

    if white_piece_amounts[WHITE_PAWN] != 0 or black_piece_amounts[WHITE_PAWN] != 0:
        return 2

    elif white_material <= PIECE_VALUES_MID[WHITE_BISHOP] and black_material <= PIECE_VALUES_MID[WHITE_BISHOP]:
        return 0

    elif white_material <= 2 * PIECE_VALUES_MID[WHITE_BISHOP] and black_material <= 2 * PIECE_VALUES_MID[WHITE_BISHOP]:

        # With only 2 knights, it's impossible to checkmate
        if white_piece_amounts[WHITE_KNIGHT] == 2 or black_piece_amounts[WHITE_KNIGHT] == 2:
            return 0

        elif white_material == 0:
            if black_material <= PIECE_VALUES_MID[WHITE_BISHOP]:
                return 0

            return 2

        elif black_material == 0:
            if white_material <= PIECE_VALUES_MID[WHITE_BISHOP]:
                return 0

            return 2

        # Here we know they both do not have 0 material, and they cannot have pawns or queens,
        # this means they either have a rook, and the other player has a minor piece,
        # or this means one player has two minor pieces, and the other players has one minor piece.

        return 1

    return 2


# @nb.njit(SCORE_TYPE(Position.class_type.instance_type), cache=True)
@nb.njit(cache=True)
def evaluate(position):

    white_mid_material = 0
    white_end_material = 0

    white_mid_scores = 0
    white_end_scores = 0

    black_mid_material = 0
    black_end_material = 0

    black_mid_scores = 0
    black_end_scores = 0

    # We make a 10 size array for each side, and eight of them are used for storing
    # the least advanced pawn. Storing this allows us to check for passed pawns,
    # backwards pawns, isolated pawns and whatnot.
    # Having a ten element array gives padding on the side to prevent out of bounds exceptions.
    pawn_rank = np.zeros((2, 10), dtype=np.uint8)

    white_piece_amounts = np.zeros(6, dtype=np.uint8)
    black_piece_amounts = np.zeros(6, dtype=np.uint8)

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
        col = i % 8 + 1

        if piece == WHITE_PAWN:
            if row < pawn_rank[0][col]:
                pawn_rank[0][col] = row

    for pos in position.black_pieces:
        piece = board[pos]
        i = MAILBOX_TO_STANDARD[pos]
        row = 8 - i // 8
        col = i % 8 + 1

        if piece == BLACK_PAWN:
            if row > pawn_rank[1][col]:
                pawn_rank[1][col] = row

    for pos in position.white_pieces:

        piece = board[pos]

        game_phase += GAME_PHASE_SCORES[piece]

        white_mid_material += PIECE_VALUES_MID[piece]
        white_end_material += PIECE_VALUES_END[piece]

        white_piece_amounts[piece] += 1

        if piece == WHITE_PAWN:
            scores = evaluate_pawn(position, pos, pawn_rank)

        elif piece == WHITE_KNIGHT:
            scores = evaluate_knight(position, pos)

        elif piece == WHITE_BISHOP:
            scores = evaluate_bishop(position, pos)

        elif piece == WHITE_ROOK:
            scores = evaluate_rook(position, pos, pawn_rank)

        elif piece == WHITE_QUEEN:
            scores = evaluate_queen(position, pos, pawn_rank)

        else:
            scores = evaluate_king(position, pos, pawn_rank)

        white_mid_scores += scores[0]
        white_end_scores += scores[1]

    for pos in position.black_pieces:

        piece = board[pos]

        game_phase += GAME_PHASE_SCORES[piece - 6]

        black_mid_material += PIECE_VALUES_MID[piece - 6]
        black_end_material += PIECE_VALUES_END[piece - 6]

        black_piece_amounts[piece - 6] += 1

        if piece == BLACK_PAWN:
            scores = evaluate_pawn(position, pos, pawn_rank)

        elif piece == BLACK_KNIGHT:
            scores = evaluate_knight(position, pos)

        elif piece == BLACK_BISHOP:
            scores = evaluate_bishop(position, pos)

        elif piece == BLACK_ROOK:
            scores = evaluate_rook(position, pos, pawn_rank)

        elif piece == BLACK_QUEEN:
            scores = evaluate_queen(position, pos, pawn_rank)

        else:
            scores = evaluate_king(position, pos, pawn_rank)

        black_mid_scores += scores[0]
        black_end_scores += scores[1]

    if white_piece_amounts[WHITE_BISHOP] >= 2:
        white_mid_scores += BISHOP_PAIR_BONUS_MID
        white_end_scores += BISHOP_PAIR_BONUS_END

    if black_piece_amounts[WHITE_BISHOP] >= 2:
        black_mid_scores += BISHOP_PAIR_BONUS_MID
        black_end_scores += BISHOP_PAIR_BONUS_END

    # If our opponent's score is already very good, and our king position is still substandard,
    # then we should be penalized. Set a cap on this at 20,
    # so we can't gain too much bonus if our king position is good.
    # If our opponent's score is so bad that adding 200 to it doesn't make it go near the positives,
    # then we should cap the minimum at -50 to avoid weird insane penalties

    white_king_penalty = min(max(-50, (black_mid_scores + 300)) *
                                   (KING_PST_MID[MAILBOX_TO_STANDARD[position.king_positions[0]]] - 10) / 200,
                             20)
    black_king_penalty = min(max(-50, (white_mid_scores + 300)) *
                                   (KING_PST_MID[MAILBOX_TO_STANDARD[position.king_positions[1]] ^ 56] - 10) / 200,
                             20)

    # print(make_readable_board(position))
    # print(white_king_penalty, black_king_penalty, white_mid_scores, black_mid_scores,
    #       KING_PST_MID[MAILBOX_TO_STANDARD[position.king_positions[0]]] - 10,
    #       KING_PST_MID[MAILBOX_TO_STANDARD[position.king_positions[1]] ^ 56] - 10)

    white_mid_scores += white_king_penalty
    black_mid_scores += black_king_penalty

    # print(white_mid_scores, black_mid_scores, white_mid_material, black_mid_material, game_phase)

    game_phase = min(game_phase, 24)  # in case of promotions

    white_mid_scores += white_mid_material
    white_end_scores += white_end_material

    black_mid_scores += black_mid_material
    black_end_scores += black_end_material

    white_score = (white_mid_scores * game_phase +
                   (24 - game_phase) * white_end_scores) / 24

    black_score = (black_mid_scores * game_phase +
                   (24 - game_phase) * black_end_scores) / 24

    material_draw = 0.5 * check_material_draw(white_piece_amounts, black_piece_amounts,
                                              white_mid_material, black_mid_material)

    white_score *= material_draw
    black_score *= material_draw

    return SCORE_TYPE((position.side * -2 + 1) * (white_score - black_score) + TEMPO_BONUS)


# @nb.njit(SCORE_TYPE(Search.class_type.instance_type, MOVE_TYPE, MOVE_TYPE), cache=True)
@nb.njit(cache=True)
def score_move(engine, move, tt_move):

    if move == tt_move:
        return 100000

    score = 0
    # standard_from_square = MAILBOX_TO_STANDARD[get_from_square(move)]
    standard_to_square = MAILBOX_TO_STANDARD[get_to_square(move)]

    selected = get_selected(move)
    occupied = get_occupied(move)
    move_type = get_move_type(move)

    if selected < BLACK_PAWN:
        if get_is_capture(move):
            score += 10000
            score += 2 * (PIECE_VALUES_MID[occupied - BLACK_PAWN] - PIECE_VALUES_MID[selected])

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
            score += 15000 + PIECE_VALUES_MID[get_promotion_piece(move)]

        elif move_type == 2:  # Castling
            score += 1000     # Helps to incentivize castling

        elif move_type == 1:  # En Passant
            score += 2000

    else:
        if get_is_capture(move):
            score += 10000
            score += 2 * (PIECE_VALUES_MID[occupied] - PIECE_VALUES_MID[selected - BLACK_PAWN])

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
            score += 15000 + PIECE_VALUES_MID[get_promotion_piece(move) - BLACK_PAWN]

        elif move_type == 2:  # Castling
            score += 1000

        elif move_type == 1:  # En Passant
            score += 2000

    return score


# @nb.njit(SCORE_TYPE(MOVE_TYPE), cache=True)
@nb.njit(cache=True)
def score_capture(move, tt_move):

    score = 0

    if move == tt_move:
        return 100000

    selected = get_selected(move)
    occupied = get_occupied(move)

    if selected < BLACK_PAWN:
        score += PIECE_VALUES_MID[occupied - BLACK_PAWN] - PIECE_VALUES_MID[selected]

    else:
        score += PIECE_VALUES_MID[occupied] - PIECE_VALUES_MID[selected - BLACK_PAWN]

    return score
