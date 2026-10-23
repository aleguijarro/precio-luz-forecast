"""Spanish national public holidays (no regional ones), computed without external deps."""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache

FIXED_HOLIDAYS = [
    (1, 1),    # Año Nuevo
    (1, 6),    # Reyes
    (5, 1),    # Día del Trabajo
    (8, 15),   # Asunción
    (10, 12),  # Fiesta Nacional
    (11, 1),   # Todos los Santos
    (12, 6),   # Constitución
    (12, 8),   # Inmaculada
    (12, 25),  # Navidad
]


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    wd = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * wd) // 451
    month, day = divmod(h + wd - 7 * m + 114, 31)
    return date(year, month, day + 1)


@cache
def national_holidays(year: int) -> frozenset[date]:
    days = {date(year, m, d) for m, d in FIXED_HOLIDAYS}
    days.add(easter_sunday(year) - timedelta(days=2))  # Viernes Santo
    return frozenset(days)


def is_holiday(day: date) -> bool:
    return day in national_holidays(day.year)
