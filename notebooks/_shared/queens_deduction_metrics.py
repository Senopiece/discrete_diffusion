"""Metrics for recorded Queens deductions and their automatic queen cascades."""

from itertools import combinations

import numpy as np


def queen_dependency_depth(coloring, deductions):
    """Reconstruct queen-to-queen supports in a completed deduction trace.

    Explicit crossedOut deductions are supplied lemmas. For each queen,
    find earlier queens whose attacks, together with those lemmas, remove
    every alternative in its row, column, or region. Minimize the longest
    parent depth, then the number of parents. A placement requiring a rule
    beyond singleton reasoning is an opaque seed at depth one.

    This measures automatic queen-cascade depth, not full proof depth:
    dependencies inside explicit eliminations and lookahead are not recorded.
    """
    board = np.asarray(coloring)
    if board.ndim != 2 or board.shape[0] != board.shape[1] or not len(board):
        raise ValueError("Expected a nonempty square region map.")
    size = len(board)
    if not np.issubdtype(board.dtype, np.integer):
        raise ValueError("Region IDs must be integers.")
    if set(np.unique(board).tolist()) != set(range(size)):
        raise ValueError("Expected exactly N region IDs numbered 0 through N-1.")

    def field(step, name):
        return step[name] if isinstance(step, dict) else getattr(step, name)

    def attack_cells(row, col):
        cells = {(row, c) for c in range(size)} | {(r, col) for r in range(size)}
        cells.update((int(r), int(c)) for r, c in zip(*np.where(board == board[row, col])))
        cells.update(
            (r, c)
            for r in range(max(0, row - 1), min(size, row + 2))
            for c in range(max(0, col - 1), min(size, col + 2))
        )
        return cells

    explicit_crossouts = set()
    all_attacked = set()
    attacks = []
    queens = []
    elimination_steps = 0
    for step_index, step in enumerate(deductions, start=1):
        state = field(step, "state")
        cells = [tuple(map(int, cell)) for cell in field(step, "cells")]
        if any(len(cell) != 2 or not (0 <= cell[0] < size and 0 <= cell[1] < size)
               for cell in cells):
            raise ValueError(f"Invalid coordinates in deduction {step_index}.")
        if state == "crossedOut":
            if any((q["row"], q["column"]) in cells for q in queens):
                raise ValueError("The trace crosses out an already placed queen.")
            explicit_crossouts.update(cells)
            elimination_steps += 1
            continue
        if state != "star":
            raise ValueError(f"Unknown deduction state: {state}")

        for row, col in cells:
            if (row, col) in explicit_crossouts or (row, col) in all_attacked:
                raise ValueError(f"Queen at {(row, col)} conflicts with previous deductions.")
            domains = [
                ("row", row, {(row, c) for c in range(size)}),
                ("column", col, {(r, col) for r in range(size)}),
                ("region", int(board[row, col]), {
                    (int(r), int(c)) for r, c in zip(*np.where(board == board[row, col]))
                }),
            ]
            best = None
            # At most 2**6 subsets for a 7x7 puzzle's seventh queen.
            for count in range(len(queens) + 1):
                for parents in combinations(range(len(queens)), count):
                    parent_depth = max((queens[p]["depth"] for p in parents), default=0)
                    parent_attacks = set().union(*(attacks[p] for p in parents))
                    for unit_order, (kind, unit_index, domain) in enumerate(domains):
                        alternatives = domain - {(row, col)}
                        required = alternatives - explicit_crossouts
                        if not required <= parent_attacks:
                            continue
                        key = (1 + parent_depth, count, parents, unit_order)
                        if best is None or key < best[0]:
                            best = (key, kind, unit_index, alternatives, required)
            if best is None:
                # The returned explanation is an advanced rule whose internal
                # proof is unavailable. Do not invent edges to earlier queens.
                depth, parents = 1, []
                kind, unit_index = None, None
                alternatives, required = set(), set()
                opaque_seed = True
            else:
                key, kind, unit_index, alternatives, required = best
                depth, parents = key[0], list(key[2])
                opaque_seed = False
            queen_id = len(queens)
            queens.append(dict(
                queen_id=queen_id,
                row=row, column=col, region=int(board[row, col]),
                trace_step=step_index, reason=str(field(step, "reason")),
                depth=int(depth), parents=parents,
                forced_unit=kind, forced_unit_index=unit_index,
                opaque_seed=opaque_seed,
                preceding_elimination_steps=elimination_steps,
                explicit_alternatives=sorted(alternatives & explicit_crossouts),
                automatic_alternatives=sorted(required),
            ))
            queen_attacks = attack_cells(row, col)
            attacks.append(queen_attacks)
            all_attacked.update(queen_attacks)
    if len(queens) != size:
        raise ValueError(f"Expected a completed trace with {size} queens, got {len(queens)}.")

    depths = [q["depth"] for q in queens]
    return dict(
        definition="queen_cascade_dependency_depth_v1",
        max_depth=max(depths),
        mean_depth=float(np.mean(depths)),
        mean_wait=float(np.mean([q["preceding_elimination_steps"] for q in queens])),
        queen_depths=depths,
        opaque_seed_count=sum(q["opaque_seed"] for q in queens),
        root_count=sum(not q["parents"] for q in queens),
        edges=[[parent, q["queen_id"]] for q in queens for parent in q["parents"]],
        queens=queens,
    )
