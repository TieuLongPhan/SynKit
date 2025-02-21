from typing import List, Dict, Any


def merge_dicts(
    list1: List[Dict[str, Any]],
    list2: List[Dict[str, Any]],
    key: str,
    intersection: bool = True,
) -> List[Dict[str, Any]]:
    """
    Merges two lists of dictionaries based on a specified key, with an option to
    either merge only dictionaries with matching key values (intersection) or
    all dictionaries (union).

    Parameters:
    - list1 (List[Dict[str, Any]]): The first list of dictionaries.
    - list2 (List[Dict[str, Any]]): The second list of dictionaries.
    - key (str): The key used to match and merge dictionaries from both lists.
    - intersection (bool): If True, only merge dictionaries with matching key values;
      if False, merge all dictionaries, combining those with matching key values.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries with merged contents from both
      input lists according to the specified merging strategy.
    """
    dict1 = {item[key]: item for item in list1}
    dict2 = {item[key]: item for item in list2}

    if intersection:
        # Intersection of keys: only keys present in both dictionaries are merged
        merged_list = []
        for item1 in list1:
            r_id = item1.get(key)
            if r_id in dict2:
                merged_item = {**item1, **dict2[r_id]}
                merged_list.append(merged_item)
        return merged_list
    else:
        # Union of keys: all keys from both dictionaries are merged
        merged_dict = {}
        all_keys = set(dict1) | set(dict2)
        for k in all_keys:
            if k in dict1 and k in dict2:
                merged_dict[k] = {**dict1[k], **dict2[k]}
            elif k in dict1:
                merged_dict[k] = dict1[k]
            else:
                merged_dict[k] = dict2[k]
        return list(merged_dict.values())
