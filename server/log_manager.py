"""In-memory dispatch log manager for agent external memory."""

from typing import List


class LogManager:
    """Maintains an in-memory dispatch log that resets per episode."""

    def __init__(self):
        self._entries: List[str] = []

    def reset(self) -> None:
        self._entries.clear()

    def append(self, note: str) -> None:
        """Append a note to the dispatch log."""
        self._entries.append(note)

    def get_log(self) -> str:
        """Return the full log as a markdown-like string."""
        if not self._entries:
            return "# Dispatch Log\n\n(No entries yet.)\n"
        lines = ["# Dispatch Log\n"]
        for i, entry in enumerate(self._entries, 1):
            lines.append(f"{i}. {entry}")
        lines.append("")
        return "\n".join(lines)

    def get_entries(self) -> List[str]:
        return self._entries.copy()
