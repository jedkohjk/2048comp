def AI(mat):
    
    GOAL = 0 
    # The goal set will be rounded down to the nearest power of two less than or equal to it.
    # For unlimited mode, change the goal to 0.
    # The time taken to make each move grows as the number of high tiles on the board increases, so the time taken for higher goals will increase disproportionately.
    # I tested unlimited mode by adding keys to the colour dictionaries and commenting out the display_end_game function calls in puzzle_AI.
    
    def reverse_m(matrix):
        
        return tuple(tuple(i[::-1]) for i in matrix)
    
    def transpose_m(matrix):
        
        return tuple(tuple(matrix[j][i] for j in range(len(matrix))) for i in range(len(matrix[0])))
    
    def merge(matrix):
        op = [[0 for j in range(len(matrix[0]))] for i in range(len(matrix))]
        for rown, row in enumerate(matrix):
            if row in merges:
                op[rown] = merges[row][:]
            else:
                old_prev, op[rown][0], last = row[0], row[0], 0
                for col in row[1:]:
                    if col > 1:
                        if old_prev > 1:
                            if col == old_prev == op[rown][last]:
                                op[rown][last] = col + 1
                            else:
                                last += 1
                                op[rown][last] = col
                        else:
                            op[rown][last] = col
                        old_prev = col
                merges[row] = op[rown][:]
        op = tuple(tuple(col for col in row) for row in op)
        return (op, op != matrix)
    
    def abs_forcer(bit_mat):
        
        def force_merge(matrix):
            
            if matrix not in forced_board:
                forced_matrix = [[0 for col in range(len(matrix[0]))] for row in range(len(matrix))]
                present_score, removed_score, removed_count = 0, 0, 0
                for rown, row in enumerate(matrix):
                    if row in forced_row:
                        forced_matrix[rown] = forced_row[row][0][:]
                        present_score += forced_row[row][1]
                        removed_score += forced_row[row][2]
                        removed_count += forced_row[row][3]
                    else:
                        old_prev, last, row_score = 0 , -1, 0
                        for coln, col in enumerate(row):
                            if col <= 1:
                                row_removed = sum(i for i in row[coln + 1:] if i > 1)
                                row_count = sum(i > 1 for i in row[coln + 1:])
                                break
                            if col == old_prev == forced_matrix[rown][last]:
                                if col >= prelim:
                                    forced_board[matrix] = (-1, 0)
                                    return forced_board[matrix]
                                forced_matrix[rown][last] += 1
                            else:
                                last += 1
                                forced_matrix[rown][last] = col
                            old_prev = col
                        else:
                            row_removed = 0
                            row_count = 0
                        row_score = sum(forced_matrix[rown])
                        present_score += row_score
                        removed_score += row_removed
                        removed_count += row_count
                        forced_row[row] = [forced_matrix[rown][:], row_score, row_removed, row_count]
                if all(((col <= 1) or (col == matrix[rown][coln])) for rown, row in enumerate(forced_matrix) for coln, col in enumerate(row)):
                    # returns a score and the number of empty squares (used only at the start to determine board complexity)
                    # the penalty score is the sum of logbase2 of all tiles on the board
                    # this rewards merging because if two tiles of values n merge into a tile (n+1), the penalty score decreases by (n-1)
                    forced_board[matrix] = (present_score + removed_score, sum(col <= 1 for row in matrix for col in row))
                else:
                    recursed_score, recursed_count = abs_forcer(forced_matrix)
                    if recursed_score <= 0:
                        forced_board[matrix] = (recursed_score - 1, 0) # if a win is guaranteed, favour the longest path to victory so as to maximise the average score metric
                    else:
                        forced_board[matrix] = (recursed_score + removed_score, recursed_count - removed_count)
            return forced_board[matrix]
                
        return min(force_merge(tuple(tuple(col for col in row) for row in bit_mat)), force_merge(reverse_m(bit_mat)), force_merge(transpose_m(bit_mat)), force_merge(reverse_m(transpose_m(bit_mat))), key = lambda i: i[0])

    def abs_pos(bit_mat):
        # between every two numbers in the grid, divides the higher number between them by their distance and squares it, and adds it to the penalty score
        # thus, tiles will get a penalty reduction by being closer to large tiles, and relatively large tiles benefit more from this penalty reduction, encouraging similar tiles to be closer together
        # this also penalises centralised tiles more as they are closer to more tiles (and hence the distance the penalty is divided by is usually smaller), encouraging larger tiles to be at the corners or edges
        # being in corners or edges means that larger tiles will not get in the way of the movement of smaller tiles (which merge more often and hence need to move more often)

        flattened_matrix = [[col, (rown, coln)] for rown, row in enumerate(bit_mat) for coln, col in enumerate(row)]
        return sum((max(num1, num2)[0] / (abs(num1[1][0] - num2[1][0]) + abs(num1[1][1] - num2[1][1]))) ** 2 for start, num1 in enumerate(flattened_matrix, 1) for num2 in flattened_matrix[start:])

    def make_moves(bit_mat, depth):
        merges = [[lambda bit_mat: bit_mat, lambda bit_mat: bit_mat, 'a'], [lambda bit_mat: reverse_m(bit_mat), lambda bit_mat: reverse_m(bit_mat), 'd'], [lambda bit_mat: transpose_m(bit_mat), lambda bit_mat: transpose_m(bit_mat), 'w'], [lambda bit_mat: reverse_m(transpose_m(bit_mat)), lambda bit_mat: transpose_m(reverse_m(bit_mat)), 's']]
        candidates = []
        has_won = False
        for move in merges:
            test_move, validity = merge(move[0](bit_mat))
            if validity:
                test_move = move[1](test_move)
                if test_move in static:
                    candidates.append(['a', test_move, static[test_move]])
                else:
                    if any(col >= win_condition for row in test_move for col in row):
                        candidates.append([move[2], test_move, [0, 0, (0, 0)]])
                    else:
                        candidates.append([move[2], test_move, [0, 0, abs_forcer(test_move)]])
                if candidates[-1][2][2][0] <= 0:
                    has_won = True
                    static[test_move] = [candidates[-1][2][2][0] - depth, 0, candidates[-1][2][2]]
        if has_won:
            return (candidates, True)
        for move in candidates:
            if move[1] not in static:
                move[2][1] = abs_pos(move[1])
        return (candidates, False)

    def main_function(bit_mat, candidates, has_won, depth):
        
        if not(len(candidates)):
            pre_move[bit_mat] = [None, [True, lose_score + 2 * depth + sum(col for row in bit_mat for col in row if col >= 1)]] # if a loss is unavoidable, losses with more merges will be favoured to maximise the average score metric, otherwise, the addition should be negligibly small in the grand scheme of things
            return pre_move[bit_mat]
        if has_won:
            best_move = min(candidates, key = lambda i: i[2])
            pre_move[bit_mat] = [best_move[0], [False, static[best_move[1]][0]]]
            return pre_move[bit_mat]
        for move in candidates:
            if move[1] not in static:
                move[2][0] = move[2][1] * (move[2][2][0] + stabiliser) # multiplies the two penalty scores derived in abs_pos and abs_forcer to give a final penalty score
                # multiplication is likely not the best way to do it; it is more important to balance the penalty scores and lower the higher metric but with multiplication, lowering already low penalty scores is prioritised as it is easier to get a high percentage change there
                # the stabiliser is used to mitigate this limitation, but it is an imperfect method
                static[move[1]] = move[2]
            else:
                move[2] = static[move[1]]
        if depth:
            candidates.sort(key = lambda i: i[2])
            test_index = 0
            best_move = [None, [True, impossible_score]]
            cut_off = impossible_score
            branch_factor = 1 + ((4 * depth) / (board_complexity * max(i[2][2][1] for i in candidates)))
            while (test_index < len(candidates)) and (best_move[1][0] or (candidates[test_index][2][0] <= cut_off)): # if there is a way to lose by force, keep looking
                if candidates[test_index][1] not in post_move:
                    depth_scores = []
                    for rown, row in enumerate(candidates[test_index][1]):
                        for coln, col in enumerate(row):
                            if col <= 1:
                                test = tuple(rw if rwn != rown else tuple(cl if cln != coln else 2 for cln, cl in enumerate(row)) for rwn, rw in enumerate(candidates[test_index][1]))
                                if test not in pre_move:
                                    pre_move[test] = main_function(test, *make_moves(test, 1 + board_complexity - depth), depth - 1)
                                depth_scores.append(pre_move[test][1])
                    post_move[candidates[test_index][1]] = [any(i[0] for i in depth_scores), (sum(i[1] for i in depth_scores) / len(depth_scores))] # prioritises having no immediate way to lose, then uses a pessimistic weighted average score
                if post_move[candidates[test_index][1]] < best_move[1]:
                    best_move = [candidates[test_index][0], post_move[candidates[test_index][1]]]
                    cut_off = branch_factor * candidates[test_index][2][0] # the search net for promising moves is cast wider initially, but is less important when further from the current position
                test_index += 1
            pre_move[bit_mat] = best_move
        else:
            best_move = min(candidates, key = lambda i: i[2])
            pre_move[bit_mat] = [best_move[0], [False, best_move[2][0] * best_move[2][0]]] # squares the scores so that when taking the average should recursion have occurred, higher scores will have a higher weightage, hence making the programme more pessimistic and focus on surviving
        return pre_move[bit_mat]
    
    board_size = len(mat) * len(mat[0])
    win_condition = int(GOAL).bit_length() # for the rest of the code, the logbase2 (or bit length) will be used instead to  more accurately denote the number of steps away from victory
    if win_condition <= 2 or win_condition > board_size + 1:
        win_condition = board_size + 2
    prelim = win_condition - 1
    lose_score = ((1 + (2 * win_condition * (1 + board_size))) ** 8) + 1 # the score is based on a penalty system; guaranteed wins are <= 0 and losses are impossibly large
    impossible_score = 2 * (lose_score + (win_condition * board_size) + 1)
    merges, pre_move, post_move, static, forced_board, forced_row = {}, {}, {}, {}, {}, {} # dictionaries used for speed improvement since there was no space limitation; these are local dictionaries and follow the rules, existing only within each call
    bm = tuple(tuple(col.bit_length() for col in row) for row in mat) # non-mutable tuples are used as they are compatible with dictionaries
    initial_cands, won = make_moves(bm, 0)
    if len(initial_cands) == 1: # Only 1 valid move
        return initial_cands[0][0]
    initial_stats = [min(i[2][2][0] for i in initial_cands), max(i[2][2][1] for i in initial_cands), min(i[2][1] for i in initial_cands)]
    board_complexity = (int(initial_stats[0] * initial_stats[2] / ((board_size * initial_stats[1]) ** 2)).bit_length()) // 2 if initial_stats[0] > 0 else 0 # determines the depth searched, prioritising surviving; the closer one is to losing (i.e. higher score and fewer blank squares), the deeper the search
    #board_complexity = 0 #uncomment this line to test evaluation
    stabiliser = sum(col * col for row in bm for col in row if col > 1) * ((initial_stats[1] / board_size) ** 2) * 3 # the stabiliser is lower when there are more tiles on the board, representing an increased importance of merging when there are fewer tiles; it also increases in tandem with the sum of squares of tiles on the board, so that what is considered a 'noteworthy merge of larger tiles' is scored relative to the board situation
    return main_function(bm, initial_cands, won, board_complexity)[0]
