from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chess


PIECE_NAME_TO_TYPE = {
    "PAWN": chess.PAWN,
    "KNIGHT": chess.KNIGHT,
    "BISHOP": chess.BISHOP,
    "ROOK": chess.ROOK,
    "QUEEN": chess.QUEEN,
    "KING": chess.KING,
}

PIECE_TYPE_TO_NAME = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

COMMAND_CHECKMATE = "CHECKMATE"
COMMAND_GAME_OVER = "GAME OVER"
COMMAND_RESIGN = "RESIGN"
EXIT_COMMANDS = {COMMAND_CHECKMATE, COMMAND_GAME_OVER, COMMAND_RESIGN}


@dataclass
class GameResult:
    finished: bool
    result_text: str | None = None
    is_loss: bool = False
    reason: str | None = None
    record_for_adaptation: bool = False


class SimpleANN:
    def __init__(self, input_size: int, hidden_size: int = 32, seed: int = 4630) -> None:
        random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w1 = [
            [random.uniform(-0.12, 0.12) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [random.uniform(-0.12, 0.12) for _ in range(hidden_size)]
        self.b2 = 0.0

    def predict(self, features: list[float]) -> float:
        hidden, _ = self._forward(features)
        output = self._sigmoid(sum(weight * value for weight, value in zip(self.w2, hidden)) + self.b2)
        return output

    def train(self, examples: list[tuple[list[float], float]], epochs: int = 8, learning_rate: float = 0.04) -> None:
        if not examples:
            return

        for _ in range(epochs):
            random.shuffle(examples)
            for features, target in examples:
                hidden, hidden_raw = self._forward(features)
                output_raw = sum(weight * value for weight, value in zip(self.w2, hidden)) + self.b2
                output = self._sigmoid(output_raw)
                delta_output = (output - target) * output * (1.0 - output)

                previous_w2 = self.w2[:]
                for index in range(self.hidden_size):
                    self.w2[index] -= learning_rate * delta_output * hidden[index]
                self.b2 -= learning_rate * delta_output

                for hidden_index in range(self.hidden_size):
                    activation_grad = 1.0 - math.tanh(hidden_raw[hidden_index]) ** 2
                    delta_hidden = delta_output * previous_w2[hidden_index] * activation_grad
                    for feature_index in range(self.input_size):
                        self.w1[hidden_index][feature_index] -= learning_rate * delta_hidden * features[feature_index]
                    self.b1[hidden_index] -= learning_rate * delta_hidden

    def _forward(self, features: list[float]) -> tuple[list[float], list[float]]:
        hidden_raw = [
            sum(weight * value for weight, value in zip(row, features)) + bias
            for row, bias in zip(self.w1, self.b1)
        ]
        hidden = [math.tanh(value) for value in hidden_raw]
        return hidden, hidden_raw

    @staticmethod
    def _sigmoid(value: float) -> float:
        clamped = max(min(value, 40.0), -40.0)
        return 1.0 / (1.0 + math.exp(-clamped))


class LossArchive:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"games": []}, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            data = {"games": []}
        data.setdefault("games", [])
        return data

    def save(self, data: dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def record_loss(self, game_record: dict[str, Any]) -> None:
        data = self.load()
        data["games"].append(game_record)
        self.save(data)


class ChessAdvisor:
    def __init__(self, player_color: chess.Color, archive_path: Path) -> None:
        self.player_color = player_color
        self.opponent_color = not player_color
        self.board = chess.Board()
        self.archive = LossArchive(archive_path)
        self.model = SimpleANN(input_size=78)
        self.game_history: list[dict[str, Any]] = []
        self.recommendation_history: list[dict[str, Any]] = []
        self.turn_number: int = 0
        self._train_from_losses()

    def run(self) -> None:
        print("Chess Assistant is ready.")
        print(f"You are playing as {'White' if self.player_color == chess.WHITE else 'Black'}.")
        print("Type CHECKMATE, GAME OVER, or RESIGN at any prompt to stop.")
        print()
        self.render_board()

        if self.player_color == chess.WHITE:
            self.turn_number += 1
            print(f"--- Turn {self.turn_number} ---")
            print("White goes first, so the assistant will recommend your opening move now.")
            result = self.recommend_and_apply_move()
            if result.finished:
                self._finish(result)
                return

        while True:
            if self.player_color == chess.BLACK:
                self.turn_number += 1
                print(f"--- Turn {self.turn_number} ---")

            result = self.prompt_and_apply_opponent_move()
            if result.finished:
                self._finish(result)
                return

            if self.player_color == chess.WHITE:
                self.turn_number += 1
                print(f"--- Turn {self.turn_number} ---")

            result = self.recommend_and_apply_move()
            if result.finished:
                self._finish(result)
                return

    def render_board(self) -> None:
        print()
        for rank in range(7, -1, -1):
            line = [str(rank + 1)]
            for file_index in range(8):
                square = chess.square(file_index, rank)
                piece = self.board.piece_at(square)
                line.append(piece.symbol() if piece else ".")
            print(" ".join(line))
        print("  a b c d e f g h")
        print()

    def prompt_and_apply_opponent_move(self) -> GameResult:
        print("Enter your opponent's move.")
        while True:
            piece_input = self._read_piece_name()
            if isinstance(piece_input, GameResult):
                return piece_input

            from_square = self._read_square("Position A")
            if isinstance(from_square, GameResult):
                return from_square

            to_square = self._read_square("Position B")
            if isinstance(to_square, GameResult):
                return to_square

            move = self._build_validated_move(piece_input, from_square, to_square, self.opponent_color)
            if move is None:
                print("Illegal move for the current board state. Please try again.")
                continue

            return self._apply_move(move, actor="opponent")

    def recommend_and_apply_move(self) -> GameResult:
        best_move = self._choose_best_move(self.player_color)
        if best_move is None:
            if self.board.is_checkmate():
                return GameResult(True, "CHECKMATE YOU LOST", is_loss=True, reason="checkmate")
            return GameResult(True, "GAME OVER", is_loss=False, reason="stalemate")

        piece_name = self._piece_name_from_square(best_move.from_square)
        print(
            f"Move {piece_name} from {chess.square_name(best_move.from_square)} to {chess.square_name(best_move.to_square)}"
        )
        apply_result = self._prompt_recommendation_decision(best_move)
        if apply_result is not None:
            return apply_result

        self.recommendation_history.append(
            {
                "fen": self.board.fen(),
                "move": best_move.uci(),
                "player_color": "white" if self.player_color == chess.WHITE else "black",
            }
        )
        return self._apply_move(best_move, actor="player")

    def prompt_and_apply_player_move(self) -> GameResult:
        print("Enter your move.")
        while True:
            piece_input = self._read_piece_name()
            if isinstance(piece_input, GameResult):
                return piece_input

            from_square = self._read_square("Position A")
            if isinstance(from_square, GameResult):
                return from_square

            to_square = self._read_square("Position B")
            if isinstance(to_square, GameResult):
                return to_square

            move = self._build_validated_move(piece_input, from_square, to_square, self.player_color)
            if move is None:
                print("Illegal move for the current board state. Please try again.")
                continue

            return self._apply_move(move, actor="player")

    def _prompt_recommendation_decision(self, recommended_move: chess.Move) -> GameResult | None:
        while True:
            choice = input("Apply recommended move? (yes/no): ").strip()
            command_result = self._command_from_input(choice)
            if command_result:
                return command_result

            normalized = choice.lower()
            if normalized in {"", "y", "yes"}:
                return None
            if normalized in {"n", "no"}:
                return self.prompt_and_apply_player_move()
            print("Please enter yes or no.")

    def _apply_move(self, move: chess.Move, actor: str) -> GameResult:
        moving_piece = self.board.piece_at(move.from_square)
        captured_piece = self._captured_piece_for_move(move)
        before_fen = self.board.fen()
        self.board.push(move)

        self.game_history.append(
            {
                "actor": actor,
                "piece": PIECE_TYPE_TO_NAME.get(moving_piece.piece_type, "Piece") if moving_piece else "Piece",
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "uci": move.uci(),
                "fen_before": before_fen,
                "fen_after": self.board.fen(),
                "capture": PIECE_TYPE_TO_NAME.get(captured_piece.piece_type) if captured_piece else None,
            }
        )

        if captured_piece and moving_piece:
            print(
                f"{PIECE_TYPE_TO_NAME[captured_piece.piece_type]} defeated by {PIECE_TYPE_TO_NAME[moving_piece.piece_type]}"
            )

        self.render_board()

        if self.board.is_check():
            print("Check")

        if self.board.is_repetition(3):
            return GameResult(
                True,
                "GAME OVER - DRAW BY REPETITION",
                is_loss=False,
                reason="threefold-repetition",
                record_for_adaptation=True,
            )

        if self.board.is_checkmate():
            if self.board.turn == self.player_color:
                return GameResult(True, "CHECKMATE YOU LOST", is_loss=True, reason="checkmate")
            return GameResult(True, "CHECKMATE YOU WON", is_loss=False, reason="checkmate")

        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
            return GameResult(True, "GAME OVER", is_loss=False, reason="draw")

        return GameResult(False)

    def _captured_piece_for_move(self, move: chess.Move) -> chess.Piece | None:
        if not self.board.is_capture(move):
            return None
        if self.board.is_en_passant(move):
            offset = -8 if self.board.turn == chess.WHITE else 8
            captured_square = move.to_square + offset
            return self.board.piece_at(captured_square)
        return self.board.piece_at(move.to_square)

    def _build_validated_move(
        self,
        piece_name: str,
        from_square_name: str,
        to_square_name: str,
        color: chess.Color,
    ) -> chess.Move | None:
        from_square = chess.parse_square(from_square_name)
        to_square = chess.parse_square(to_square_name)
        piece = self.board.piece_at(from_square)

        if piece is None or piece.color != color:
            print("No matching piece exists on Position A. Please try again.")
            return None

        if piece.piece_type != PIECE_NAME_TO_TYPE[piece_name]:
            print("That piece name does not match the board. Please try again.")
            return None

        promotion = chess.QUEEN if piece.piece_type == chess.PAWN and chess.square_rank(to_square) in {0, 7} else None
        move = chess.Move(from_square, to_square, promotion=promotion)
        if move in self.board.legal_moves:
            return move

        return None

    def _choose_best_move(self, color: chess.Color) -> chess.Move | None:
        if self.board.turn != color:
            return None

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        best_score = float("-inf")
        best_move = legal_moves[0]

        for move in legal_moves:
            heuristic = self._heuristic_score(move)
            ann_score = self.model.predict(self._encode_features(self.board, move, color))
            repetition_penalty = self._loss_penalty(self.board.fen(), move.uci())
            total_score = heuristic + (ann_score * 1.5) - repetition_penalty
            if total_score > best_score:
                best_score = total_score
                best_move = move

        return best_move

    def _heuristic_score(self, move: chess.Move) -> float:
        moving_piece = self.board.piece_at(move.from_square)
        captured_piece = self._captured_piece_for_move(move)
        score = 0.0

        if captured_piece:
            score += PIECE_VALUES[captured_piece.piece_type] + 0.2
        if moving_piece and moving_piece.piece_type in {chess.KNIGHT, chess.BISHOP} and chess.square_rank(move.from_square) in {0, 7}:
            score += 0.25
        if moving_piece and moving_piece.piece_type == chess.PAWN and chess.square_file(move.to_square) in {3, 4}:
            score += 0.15
        if self.board.is_castling(move):
            score += 0.6
        if move.promotion == chess.QUEEN:
            score += 4.5

        trial_board = self.board.copy(stack=False)
        trial_board.push(move)
        score += self._material_balance(trial_board, self.player_color) * 0.08
        if trial_board.is_check():
            score += 0.75
        if trial_board.is_checkmate():
            score += 100.0

        destination_attackers = trial_board.attackers(not self.player_color, move.to_square)
        destination_defenders = trial_board.attackers(self.player_color, move.to_square)
        if destination_attackers and not destination_defenders:
            score -= PIECE_VALUES.get(moving_piece.piece_type, 0.0) * 0.35 if moving_piece else 0.0

        return score

    def _material_balance(self, board: chess.Board, perspective: chess.Color) -> float:
        balance = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = PIECE_VALUES[piece.piece_type]
            balance += value if piece.color == perspective else -value
        return balance

    def _encode_features(self, board: chess.Board, move: chess.Move, perspective: chess.Color) -> list[float]:
        features: list[float] = []
        rotate = perspective == chess.BLACK

        for square in chess.SQUARES:
            mapped_square = chess.square_mirror(square) if rotate else square
            piece = board.piece_at(mapped_square)
            if piece is None:
                features.append(0.0)
                continue
            value = piece.piece_type / 6.0
            features.append(value if piece.color == perspective else -value)

        from_square = chess.square_mirror(move.from_square) if rotate else move.from_square
        to_square = chess.square_mirror(move.to_square) if rotate else move.to_square
        moving_piece = board.piece_at(move.from_square)
        captured_piece = self._captured_piece_for_move(move)

        features.extend(
            [
                1.0,
                from_square / 63.0,
                to_square / 63.0,
                chess.square_file(from_square) / 7.0,
                chess.square_rank(from_square) / 7.0,
                chess.square_file(to_square) / 7.0,
                chess.square_rank(to_square) / 7.0,
                (moving_piece.piece_type / 6.0) if moving_piece else 0.0,
                (captured_piece.piece_type / 6.0) if captured_piece else 0.0,
            ]
        )
        features.append(1.0 if board.is_castling(move) else 0.0)
        features.append(1.0 if board.is_capture(move) else 0.0)
        features.append((move.promotion or 0) / 6.0)

        trial_board = board.copy(stack=False)
        trial_board.push(move)
        features.append(1.0 if trial_board.is_check() else 0.0)
        features.append(1.0 if trial_board.is_checkmate() else 0.0)
        return features

    def _loss_penalty(self, fen: str, move_uci: str) -> float:
        data = self.archive.load()
        penalty = 0.0
        for game in data.get("games", []):
            for recommendation in game.get("recommendations", []):
                if recommendation.get("fen") == fen and recommendation.get("move") == move_uci:
                    penalty += 0.7
        return penalty

    def _train_from_losses(self) -> None:
        data = self.archive.load()
        examples: list[tuple[list[float], float]] = []

        for game in data.get("games", []):
            color_name = game.get("player_color", "white")
            perspective = chess.WHITE if color_name == "white" else chess.BLACK
            for recommendation in game.get("recommendations", []):
                fen = recommendation.get("fen")
                move_uci = recommendation.get("move")
                if not fen or not move_uci:
                    continue
                board = chess.Board(fen)
                try:
                    bad_move = chess.Move.from_uci(move_uci)
                except ValueError:
                    continue

                if bad_move in board.legal_moves:
                    examples.append((self._encode_features(board, bad_move, perspective), 0.0))

                alternative = self._best_heuristic_move(board, perspective, excluded_uci=move_uci)
                if alternative is not None:
                    examples.append((self._encode_features(board, alternative, perspective), 1.0))

        self.model.train(examples)

    def _best_heuristic_move(self, board: chess.Board, perspective: chess.Color, excluded_uci: str) -> chess.Move | None:
        best_score = float("-inf")
        best_move = None
        current_board = self.board
        self.board = board
        try:
            for move in board.legal_moves:
                if move.uci() == excluded_uci:
                    continue
                score = self._heuristic_score(move)
                if score > best_score:
                    best_score = score
                    best_move = move
        finally:
            self.board = current_board
        return best_move

    def _piece_name_from_square(self, square: chess.Square) -> str:
        piece = self.board.piece_at(square)
        return PIECE_TYPE_TO_NAME.get(piece.piece_type, "Piece") if piece else "Piece"

    def _read_piece_name(self) -> str | GameResult:
        while True:
            raw_value = input("Piece: ").strip()
            command_result = self._command_from_input(raw_value)
            if command_result:
                return command_result

            normalized = raw_value.upper()
            if normalized in PIECE_NAME_TO_TYPE:
                return normalized

            print("Invalid Piece/ Mispelled, please try again")

    def _read_square(self, label: str) -> str | GameResult:
        while True:
            raw_value = input(f"{label}: ").strip().lower()
            command_result = self._command_from_input(raw_value)
            if command_result:
                return command_result

            if len(raw_value) == 2 and raw_value[0] in "abcdefgh" and raw_value[1] in "12345678":
                return raw_value

            print("Invalid board position, please try again")

    def _command_from_input(self, raw_value: str) -> GameResult | None:
        normalized = raw_value.strip().upper()
        if normalized == COMMAND_RESIGN:
            return GameResult(True, "GAME OVER", is_loss=True, reason="resigned")
        if normalized == COMMAND_CHECKMATE:
            return GameResult(True, "CHECKMATE", is_loss=False, reason="manual-stop")
        if normalized == COMMAND_GAME_OVER:
            return GameResult(True, "GAME OVER", is_loss=False, reason="manual-stop")
        return None

    def _finish(self, result: GameResult) -> None:
        print(result.result_text)
        if result.is_loss or result.record_for_adaptation:
            self.archive.record_loss(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "player_color": "white" if self.player_color == chess.WHITE else "black",
                    "reason": result.reason,
                    "moves": self.game_history,
                    "recommendations": self.recommendation_history,
                }
            )
            print(f"Game recorded in {self.archive.path.name} for future adaptation.")


def prompt_player_color() -> chess.Color:
    while True:
        choice = input("Choose your color (white/black): ").strip().lower()
        if choice in {"white", "w"}:
            return chess.WHITE
        if choice in {"black", "b"}:
            return chess.BLACK
        print("Please enter white or black.")


def main() -> None:
    color = prompt_player_color()
    advisor = ChessAdvisor(
        player_color=color,
        archive_path=Path(__file__).resolve().with_name("losses.json"),
    )
    advisor.run()