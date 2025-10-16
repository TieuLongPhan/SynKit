from typing import List, Dict, Any, Optional
import joblib
from tqdm import tqdm as _tqdm_type


class TqdmJoblib:
    """
    Context manager to integrate a tqdm progress bar with joblib's Parallel.

    Usage:
        with TqdmJoblib(tqdm(total=100)) as t:
            Parallel(...)(delayed(...)(...) for ...)

    :param tqdm_obj: an instantiated tqdm progress bar object (e.g. `tqdm(total=...)`).
    :type tqdm_obj: tqdm.tqdm
    :param max_batch_size: maximum sensible batch size to accept from joblib's
                           callback; anything larger will be treated as 1 to avoid
                           corrupting the bar.
    :type max_batch_size: int, optional
    :param allow_overshoot: if False (default) cap updates so the bar never
                            exceeds `tqdm_obj.total`. If True, allow normal updates.
    :type allow_overshoot: bool, optional
    :param silent_close_errors: swallow exceptions raised when closing the tqdm
                                object on exit (default True).
    :type silent_close_errors: bool, optional
    :raises AttributeError: if joblib has no `parallel.BatchCompletionCallBack`.
    """

    def __init__(
        self,
        tqdm_obj: _tqdm_type,
        max_batch_size: int = 10**6,
        allow_overshoot: bool = False,
        silent_close_errors: bool = True,
    ) -> None:
        self.tqdm_obj = tqdm_obj
        self.max_batch_size = int(max_batch_size)
        self.allow_overshoot = bool(allow_overshoot)
        self.silent_close_errors = bool(silent_close_errors)

        self._orig: Optional[Any] = None
        self._patched = False

    def __enter__(self):
        orig_cb = getattr(joblib.parallel, "BatchCompletionCallBack", None)
        if orig_cb is None:
            raise AttributeError("joblib.parallel.BatchCompletionCallBack not found")

        self._orig = orig_cb

        def _callback(*args, **kwargs):
            try:
                n = int(args[0])
                if n <= 0 or n > self.max_batch_size:
                    raise ValueError
            except Exception:
                n = 1

            if not self.allow_overshoot:
                remaining = max(0, (self.tqdm_obj.total or 0) - (self.tqdm_obj.n or 0))
                inc = n if n <= remaining else remaining
            else:
                inc = n

            if inc > 0:
                try:
                    self.tqdm_obj.update(inc)
                except Exception:
                    # If tqdm object's update fails for any reason, fallback to +1
                    try:
                        self.tqdm_obj.update(1)
                    except Exception:
                        pass

            return self._orig(*args, **kwargs)

        joblib.parallel.BatchCompletionCallBack = _callback
        self._patched = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._patched and self._orig is not None:
            try:
                joblib.parallel.BatchCompletionCallBack = self._orig
            except Exception:
                # best-effort restore; ignore to avoid masking user exceptions
                pass

        try:
            if self.tqdm_obj is not None:
                self.tqdm_obj.close()
        except Exception:
            if not self.silent_close_errors:
                raise


def merge_dicts(
    list1: List[Dict[str, Any]],
    list2: List[Dict[str, Any]],
    key: str,
    intersection: bool = True,
) -> List[Dict[str, Any]]:
    """Merges two lists of dictionaries based on a specified key, with an
    option to either merge only dictionaries with matching key values
    (intersection) or all dictionaries (union).

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
