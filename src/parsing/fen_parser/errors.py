class FENParseError(ValueError):
    def __init__(self, message: str, source: str | None = None,
                 line_number: int | None = None) -> None:
        self.message = message
        self.source = source
        self.line_number = line_number
        super().__init__(self.__str__())

    def __str__(self) -> str:
        location = self.source or "FEN"
        if self.line_number is not None:
            location = f"{location}:{self.line_number}"
        return f"{location}: {self.message}"
