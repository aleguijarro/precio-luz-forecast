from datetime import date

from luzcast.calendar import easter_sunday, is_holiday


def test_easter_known_years():
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)


def test_national_holidays():
    assert is_holiday(date(2026, 10, 12))   # Fiesta Nacional
    assert is_holiday(date(2026, 4, 3))     # Viernes Santo
    assert not is_holiday(date(2026, 10, 13))
