import pytest

from .substring import substring_and_remap


@pytest.fixture
def s_fix():
    return """abcdefghijklmnopqrstuvwxyz""".strip()


def test_no_remove_region_whole_string(s_fix):
    s_window = (0, len(s_fix) - 1)
    remove_regions = []
    pointers = [0, 5, 7]
    expected_substring = s_fix
    expected_pointers = [0, 5, 7]

    substring, remapped_pointers = substring_and_remap(
        s_fix, s_window, remove_regions, pointers
    )
    assert substring == expected_substring
    assert remapped_pointers == expected_pointers


def test_substring_start(s_fix):
    s_window = (4, len(s_fix) - 1)
    remove_regions = []
    pointers = [4, 5, 10]
    expected_substring = "efghijklmnopqrstuvwxyz"
    expected_pointers = [0, 1, 6]

    substring, remapped_pointers = substring_and_remap(
        s_fix, s_window, remove_regions, pointers
    )
    assert substring == expected_substring
    assert remapped_pointers == expected_pointers


def test_substring_end(s_fix):
    s_window = (0, len(s_fix) - 5)
    remove_regions = []
    pointers = [0, 5, 15]
    expected_substring = "abcdefghijklmnopqrstuv"
    expected_pointers = [0, 5, 15]
    substring, remapped_pointers = substring_and_remap(
        s_fix, s_window, remove_regions, pointers
    )
    assert substring == expected_substring
    assert remapped_pointers == expected_pointers


@pytest.mark.parametrize(
    "pointers",
    (
        [3, 9, 18],
        [18, 3, 9],
    ),
)
@pytest.mark.parametrize(
    "_name, remove_region, expected_substring, expected_pointers",
    [
        ("remove_region_start", [(0, 2)], "defghijklmnopqrstuvwxyz", [0, 6, 15]),
        ("remove_region_end", [(20, 25)], "abcdefghijklmnopqrst", [3, 9, 18]),
        (
            "remove_region_middle_after_pointers",
            [(22, 24)],
            "abcdefghijklmnopqrstuvz",
            [3, 9, 18],
        ),
        (
            "remove_region_middle_in_between_pointers",
            [(4, 8)],
            "abcdjklmnopqrstuvwxyz",
            [3, 4, 13],
        ),
        (
            "remove_region_middle_in_between_pointers2",
            [(10, 17)],
            "abcdefghijstuvwxyz",
            [3, 9, 10],
        ),
        ("remove_region_multiple", [(0, 2), (20, 25)], "defghijklmnopqrst", [0, 6, 15]),
        (
            "remove_region_multiple2",
            [(0, 2), (4, 8), (20, 25)],
            "djklmnopqrst",
            [0, 1, 10],
        ),
        (
            "remove_region_multiple3",
            [(0, 2), (4, 8), (10, 17), (19, 25)],
            "djs",
            [0, 1, 2],
        ),
        ("shuffled", [(20, 25), (0, 2)], "defghijklmnopqrst", [0, 6, 15]),
        ("shuffled2", [(10, 17), (0, 2), (4, 8)], "djstuvwxyz", [0, 1, 2]),
    ],
)
def test_remove_region_no_substring(
    s_fix, _name, remove_region, expected_substring, expected_pointers, pointers
):
    s_window = (0, len(s_fix) - 1)
    # print("before pointers:", [
    #     f"{p} ({s_fix[p]})" for p in pointers
    # ])
    substring, remapped_pointers = substring_and_remap(
        s_fix, s_window, remove_region, pointers
    )
    # print("after pointers:", [
    #     f"{p} ({substring[p]})" for p in remapped_pointers if p < len(substring)
    # ])

    remapped_pointers.sort()

    assert substring == expected_substring
    assert remapped_pointers == expected_pointers
