"""Меню со стрелками для PowerShell / Windows-консоли и Unix-терминала."""

from __future__ import annotations

import sys
from typing import Sequence


def _enable_windows_vt() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _read_key() -> str:
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            mapping = {"H": "up", "P": "down", "K": "left", "M": "right"}
            return mapping.get(ch2, ch2)
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x1b":
            return "quit"
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch.lower() == "q":
            return "quit"
        return ch

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            rest = sys.stdin.read(2)
            mapping = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}
            return mapping.get(rest, "quit")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch.lower() == "q":
            return "quit"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_option(
    title: str,
    options: Sequence[str],
    *,
    hint: str = "↑/↓ выбрать · Enter подтвердить · Q выход",
) -> int:
    """Вернуть индекс выбранного пункта. KeyboardInterrupt / Q → SystemExit(0)."""
    if not options:
        raise ValueError("Список пунктов пуст")
    _enable_windows_vt()
    index = 0
    drawn = 0

    def render(first: bool) -> None:
        nonlocal drawn
        if not first and drawn:
            sys.stdout.write(f"\033[{drawn}A")
        lines = [title, hint, ""]
        for i, label in enumerate(options):
            marker = "▶" if i == index else " "
            if i == index:
                lines.append(f"  \033[7m {marker} {label} \033[0m")
            else:
                lines.append(f"    {marker} {label}")
        text = "\n".join(lines) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        drawn = text.count("\n")

    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        render(first=True)
        while True:
            key = _read_key()
            if key == "up":
                index = (index - 1) % len(options)
                render(first=False)
            elif key == "down":
                index = (index + 1) % len(options)
                render(first=False)
            elif key == "enter":
                sys.stdout.write("\n")
                sys.stdout.flush()
                return index
            elif key == "quit":
                raise KeyboardInterrupt
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
