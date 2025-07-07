from typing import NamedTuple


class RegionToRemoveLen(NamedTuple):
    region_end: int
    cum_sum: int


def bin_search_region(
    region_to_remove_len: list[RegionToRemoveLen], pointer: int
) -> int:
    """
    Binary search for the cumulative sum of lengths of regions to remove

    Args:
        region_to_remove_len: list[RegionToRemoveLen]: List of regions ends
            sorted ASC with their cumulative lengths removed.
        pointer: pointer to search for in the cumulative sums.

    Returns: How much to subtract from the pointer to remap it.
    """
    left, right = 0, len(region_to_remove_len) - 1

    while left <= right:
        mid = (left + right) // 2
        if region_to_remove_len[mid].region_end < pointer:
            left = mid + 1
        elif region_to_remove_len[mid].region_end > pointer:
            right = mid - 1
        else:
            raise ValueError(
                f"Pointer {pointer} is within a remove region "
                f" end {region_to_remove_len[mid].region_end}."
            )
    # here left == right
    # 1. pointer is less than first region (so left, right = 0) - no remap
    # 2. Pointer is somewhere in the middle or greater than the last region
    #    we need to return the cumulative sum of lengths of regions
    return region_to_remove_len[left - 1].cum_sum if left > 0 else 0


def substring_and_remap(
    s: str,
    s_window: tuple[int, int],
    remove_regions: list[tuple[int, int]],
    pointers: list[int],
) -> tuple[str, list[int]]:
    """
    Extracts a substring from the string `s` based on the provided window,
    removes specified regions, and remaps pointers accordingly.

    - all tuples represent [start, end] indices (inclusive both ends)
    - s_window, remove_regions, and pointers must be less than the length of s.
    - remove_regions, pointers must be less than s_window[1]
    - no pointers within a remove_region
    - no overlapping remove_regions
    - remove_regions and pointers can be unsorted

    Args:
        s (str): The input string.
        s_window (tuple[int, int]): The start and end indices of the substring.
        remove_regions (list[tuple[int, int]]): Regions to be removed from the substring.
        pointers (list[int]): List of pointers to be remapped.

    Returns:
        tuple[str, list[int]]: The modified substring and remapped pointers.
    """

    # Step 1: Extract the substring
    substring = s[s_window[0] : s_window[1] + 1]

    # step 2: calculate new indices for remove and pointers
    norm_regions = [
        ((r1 - s_window[0]), (r2 - s_window[0])) for r1, r2 in remove_regions
    ]
    # sort regions by start index
    norm_regions.sort()
    norm_pointers = [p - s_window[0] for p in pointers]

    # Step 3: Remove regions from the substring
    keep_substrings: list[str] = []
    region_to_remove_len: list[RegionToRemoveLen] = []
    substring_len = len(substring)
    last_end = 0
    removed_len = 0
    for r1, r2 in norm_regions:
        keep_substrings.append(substring[last_end:r1])
        removed_len = r2 - r1 + 1 + removed_len
        region_to_remove_len.append(RegionToRemoveLen(r2, removed_len))
        last_end = r2 + 1

    if last_end < substring_len:
        keep_substrings.append(substring[last_end:substring_len])

    substring = "".join(keep_substrings)

    # Step 4: Remap pointers
    for i, pointer in enumerate(norm_pointers):
        remapped_pointer = pointer - bin_search_region(region_to_remove_len, pointer)
        norm_pointers[i] = remapped_pointer

    return substring, norm_pointers
