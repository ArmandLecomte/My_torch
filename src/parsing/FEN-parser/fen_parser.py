import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import FENParseError
from .fen_position import FENPosition


_PIECES = frozenset("pnbrqkPNBRQK")
_CASTLING_ORDER = "KQkq"
_EXPECTED_RESULTS = {
    "Nothing",
    "Check",
    "Checkmate",
    "Check White",
    "Check Black",
    "Checkmate White",
    "Checkmate Black",
}
_EN_PASSANT_PATTERN = re.compile(r"^[a-h][36]$")
_INTEGER_PATTERN = re.compile(r"^[0-9]+$")

_EMPTY_COUNTS = "12345678"
_BOARD_SIZE = 8


@dataclass(frozen=True)
class FENRecord:
    """A position read from a file and its optional expected analyzer result."""

    position: FENPosition
    expected_result: str | None = None


def _split_rows(placement: str) -> list[str]:
    rows = placement.split("/")
    if len(rows) != _BOARD_SIZE:
        raise FENParseError(
            f"piece placement must contain {_BOARD_SIZE} rows, got {len(rows)}"
        )
    return rows


def _parse_character(character: str, rank: int) -> list[str | None]:
    if character in _EMPTY_COUNTS:
        return [None] * int(character)
    if character in _PIECES:
        return [character]
    raise FENParseError(f"invalid character {character} in row {rank}")


def _validate_active_color(active_color: str) -> None:
    if active_color not in {"w", "b"}:
        raise FENParseError("active color must be 'w' or 'b'")


def _validate_castling(castling: str) -> None:
    if castling == "-":
        return
    if (not castling or any(right not in _CASTLING_ORDER for right in castling)
            or len(set(castling)) != len(castling)):
        raise FENParseError("castling rights must be '-', or unique letters from KQkq")
    if castling != "".join(right for right in _CASTLING_ORDER if right in castling):
        raise FENParseError("castling rights must appear in KQkq order")


def _validate_en_passant(en_passant: str, active_color: str) -> None:
    if en_passant == "-":
        return
    if not _EN_PASSANT_PATTERN.fullmatch(en_passant):
        raise FENParseError("en passant target must be '-', or a square on rank 3 or 6")
    expected_rank = "6" if active_color == "w" else "3"
    if en_passant[1] != expected_rank:
        raise FENParseError(
            f"en passant target rank must be {expected_rank} for active color {active_color}"
        )


def _parse_integer(value: str, field_name: str, minimum: int) -> int:
    if not _INTEGER_PATTERN.fullmatch(value):
        raise FENParseError(f"{field_name} must be a non-negative integer")
    parsed_value = int(value)
    if parsed_value < minimum:
        raise FENParseError(f"{field_name} must be {"positive" if minimum == 1 else "non-negative"}")
    return parsed_value


def _parse_row(row: str, rank: int) -> tuple[str | None, ...]:
    squares: list[str | None] = []
    previous_was_count = False

    for character in row:
        is_count = character in _EMPTY_COUNTS
        if is_count and previous_was_count:
            raise FENParseError(
                f"row {rank} has consecutive empty-square counts"
            )

        squares.extend(_parse_character(character, rank))
        if len(squares) > _BOARD_SIZE:
            raise FENParseError(
                f"row {rank} describes more than {_BOARD_SIZE} squares"
            )
        previous_was_count = is_count

    if len(squares) != _BOARD_SIZE:
        raise FENParseError(
            f"row {rank} describes {len(squares)} squares, "
            f"expected {_BOARD_SIZE}"
        )
    return tuple(squares)


def _parse_placement(placement: str) -> tuple[tuple[str | None, ...], ...]:
    rows = _split_rows(placement)
    board = [
        _parse_row(row, rank=_BOARD_SIZE - index)
        for index, row in enumerate(rows)
    ]
    return tuple(board)


class FENParser:
    """Parse FEN strings and files containing one position per line."""

    def parse(self, fen: str) -> FENPosition:
        fen_parts = fen.split()
        if len(fen_parts) != 6:
            raise FENParseError(
                f"expected exactly 6 FEN fields, got {len(fen_parts)}"
            )

        placement, active_color, castling, en_passant, halfmove, fullmove = fen_parts
        board = _parse_placement(placement)
        _validate_active_color(active_color)
        _validate_castling(castling)
        _validate_en_passant(en_passant, active_color)
        halfmove_clock = _parse_integer(halfmove, "halfmove clock", 0)
        fullmove_number = _parse_integer(fullmove, "fullmove number", 1)

        return FENPosition(
            board=board,
            active_color=active_color,
            castling_rights=castling,
            en_passant_target=None if en_passant == "-" else en_passant,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
        )

    def parse_line(self, line: str) -> FENRecord:
        fields = line.split()
        if len(fields) < 6:
            raise FENParseError(
                f"expected a FEN position with 6 fields, got {len(fields)}"
            )

        position = self.parse(" ".join(fields[:6]))
        expected_result = " ".join(fields[6:]) or None
        if expected_result is not None and expected_result not in _EXPECTED_RESULTS:
            raise FENParseError(f"unknown expected result: {expected_result!r}")
        return FENRecord(position=position, expected_result=expected_result)

    def parse_file(self, file_path: str | Path) -> list[FENRecord]:
        """Read positions from a UTF-8 text file, skipping blank lines."""
        path = Path(file_path)
        try:
            with path.open("r", encoding="utf-8") as source_file:
                return self.parse_lines(source_file, source=str(path))
        except OSError as error:
            raise FENParseError(f"cannot read file: {error.strerror or error}",
                                source=str(path)) from error
        except UnicodeError as error:
            raise FENParseError(f"file is not valid UTF-8: {error}",
                                source=str(path)) from error

    def parse_lines(self, lines: Iterable[str],
                    source: str = "<input>") -> list[FENRecord]:
        records = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(self.parse_line(line))
            except FENParseError as error:
                raise FENParseError(error.message, source, line_number) from error
        return records
