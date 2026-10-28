from dataclasses import dataclass


@dataclass(frozen=True)
class FENPosition:
    """A validated chess position expressed by the six FEN fields."""
    board: tuple[tuple[str | None, ...], ...]
    active_color: str
    castling_rights: str
    en_passant_target: str | None
    halfmove_clock: int
    fullmove_number: int

    @property
    def placement(self) -> str:
        return "/".join(self._encode_rank(rank) for rank in self.board)

    def _encode_rank(self, rank) -> str:
        encoded = []
        empty_squares = 0
        for piece in rank:
            if piece is None:
                empty_squares += 1
                continue
            encoded.append(self._flush_empty(empty_squares))
            encoded.append(piece)
            empty_squares = 0
        encoded.append(self._flush_empty(empty_squares))
        return "".join(encoded)

    @staticmethod
    def _flush_empty(count: int) -> str:
        return str(count) if count else ""

    def piece_at(self, square: str) -> str | None:
        """Return the piece on a square such as 'e4', or None if it is empty."""
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            raise ValueError(f"invalid chess square: {square}")
        row = 8 - int(square[1])
        column = ord(square[0]) - ord("a")
        return self.board[row][column]

    def to_fen(self) -> str:
        en_passant = self.en_passant_target or "-"
        return (f"{self.placement} {self.active_color} {self.castling_rights} "
                f"{en_passant} {self.halfmove_clock} {self.fullmove_number}")
