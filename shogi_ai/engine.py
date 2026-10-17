from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .core import BLACK, PIECE_VALUES, Move, Position, promotion_zone

MATE_SCORE = 1_000_000


@dataclass
class SearchResult:
    move: Optional[Move]
    score: int
    nodes: int
    depth: int
    elapsed: float
    candidates: tuple = ()


class AlphaBetaEngine:
    def __init__(self, depth: int = 3, time_limit: float = 2.0, seed: Optional[int] = None):
        self.depth = depth
        self.time_limit = time_limit
        self.started = 0.0
        self.nodes = 0
        self.rng = random.Random(20260921 if seed is None else seed)

    def search(self, pos: Position, depth: Optional[int] = None) -> SearchResult:
        self.started = time.time()
        self.nodes = 0
        max_depth = depth or self.depth
        best_move: Optional[Move] = None
        best_score = -math.inf
        completed_depth = 0
        candidates = ()
        for d in range(1, max_depth + 1):
            score, move, scored_moves = self._root(pos, d)
            if time.time() - self.started > self.time_limit and best_move is not None:
                break
            if move is not None:
                best_score, best_move, completed_depth = score, move, d
                candidates = tuple(scored_moves)
        return SearchResult(best_move, int(best_score), self.nodes, completed_depth, time.time() - self.started, candidates)

    def _root(self, pos: Position, depth: int) -> Tuple[int, Optional[Move], list]:
        moves = self._ordered_moves(pos)
        if not moves:
            return -MATE_SCORE, None, []
        best_move = moves[0]
        alpha = -math.inf
        scored_moves = []
        for move in moves:
            child = pos.make_move(move)
            score = -self._negamax(child, depth - 1, -math.inf, -alpha)
            scored_moves.append((move, int(score)))
            if score > alpha:
                alpha, best_move = score, move
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        return int(alpha), best_move, scored_moves

    def _negamax(self, pos: Position, depth: int, alpha: float, beta: float) -> int:
        self.nodes += 1
        if time.time() - self.started > self.time_limit:
            return evaluate(pos)
        moves = pos.legal_moves()
        if not moves:
            return -MATE_SCORE + pos.move_number
        if depth <= 0:
            return quiescence(pos, alpha, beta)
        for move in self._ordered_moves(pos, moves):
            child = pos.make_move(move)
            score = -self._negamax(child, depth - 1, -beta, -alpha)
            if score >= beta:
                return int(beta)
            if score > alpha:
                alpha = score
        return int(alpha)

    def _ordered_moves(self, pos: Position, moves=None):
        moves = list(moves or pos.legal_moves())
        self.rng.shuffle(moves)
        return sorted(moves, key=lambda m: move_order_score(pos, m), reverse=True)


def quiescence(pos: Position, alpha: float, beta: float) -> int:
    stand = evaluate(pos)
    if stand >= beta:
        return int(beta)
    if stand > alpha:
        alpha = stand
    noisy = [m for m in pos.legal_moves() if pos.piece_at(m.to_sq) is not None or m.promote]
    noisy.sort(key=lambda m: move_order_score(pos, m), reverse=True)
    for move in noisy[:12]:
        score = -quiescence(pos.make_move(move), -beta, -alpha)
        if score >= beta:
            return int(beta)
        if score > alpha:
            alpha = score
    return int(alpha)


def move_order_score(pos: Position, move: Move) -> int:
    score = 0
    captured = pos.piece_at(move.to_sq)
    if captured:
        score += 10_000 + PIECE_VALUES[captured.kind]
    if move.promote:
        score += 900
    if move.drop:
        score += PIECE_VALUES.get(move.piece, 0) // 3
    return score


def evaluate(pos: Position) -> int:
    score = 0
    for r, row in enumerate(pos.board):
        for c, piece in enumerate(row):
            if not piece:
                continue
            value = PIECE_VALUES[piece.kind] + positional_bonus(piece.kind, piece.color, r, c)
            score += value if piece.color == pos.turn else -value
    for color, hand in pos.hands.items():
        sign = 1 if color == pos.turn else -1
        for kind, count in hand.items():
            score += sign * count * int(PIECE_VALUES[kind] * 0.9)
    if pos.is_in_check(pos.turn):
        score -= 120
    if pos.is_in_check(-pos.turn):
        score += 120
    return int(score)


def positional_bonus(kind: str, color: int, r: int, c: int) -> int:
    center = 4 - abs(4 - c)
    advance = (8 - r) if color == BLACK else r
    bonus = center * 4
    if kind in ("S", "N", "P"):
        bonus += advance * 3
    if kind in ("R", "B"):
        bonus += center * 8
    if kind == "K":
        bonus -= abs(4 - c) * 4
    if promotion_zone(color, r) and kind in ("P", "L", "N", "S", "B", "R"):
        bonus += 20
    return bonus

