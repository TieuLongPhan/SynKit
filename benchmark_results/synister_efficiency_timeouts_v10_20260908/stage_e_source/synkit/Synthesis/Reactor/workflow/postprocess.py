import logging
from typing import Iterable, List, Dict, Any, Callable, Optional

from joblib import Parallel, delayed
import joblib

# optional progress bar
try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm not installed
    tqdm = None

from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.utils import clean_radical_rsmi


class _TqdmJoblib:
    def __init__(self, tqdm_obj):
        self.tqdm_obj = tqdm_obj
        self._old_batch_callback = None

    def __enter__(self):
        self._old_batch_callback = joblib.parallel.BatchCompletionCallBack

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):  # type: ignore[valid-type]
            def __call__(self_inner, *args, **kwargs):
                try:
                    self.tqdm_obj.update(n=self_inner.batch_size)
                except Exception:
                    pass
                return super(TqdmBatchCompletionCallback, self_inner).__call__(
                    *args, **kwargs
                )

        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback  # type: ignore[attr-defined]
        return self.tqdm_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        joblib.parallel.BatchCompletionCallBack = self._old_batch_callback
        self.tqdm_obj.close()


class PostSyn:
    """
    Post-processing helper for reaction data: standardize reactions and clean AAM strings,
    with optional parallelism, progress reporting, and filtering of incomplete reaction SMILES
    inside fw/bw lists. Input keys for reaction, fw, and bw are configurable.
    """

    def __init__(
        self,
        n_jobs: int = 1,
        verbose: int = 2,
        standardizer: Optional[Standardize] = None,
        reaction_key: str = "reactions",
        fw_key: str = "fw",
        bw_key: str = "bw",
    ) -> None:
        """
        :param n_jobs: number of parallel jobs.
        :param verbose: verbosity level (0=errors only,1=warnings,2=info,3=debug).
        :param standardizer: optional Standardize instance.
        :param reaction_key: key in input dict holding the reaction SMILES.
        :param fw_key: key for forward AAMs/reaction SMILES list.
        :param bw_key: key for backward AAMs/reaction SMILES list.
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.std = standardizer if standardizer is not None else Standardize()
        self.reaction_key = reaction_key
        self.fw_key = fw_key
        self.bw_key = bw_key

        self.logger = logging.getLogger(self.__class__.__name__)
        self._configure_logger()

    def _configure_logger(self):
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        if self.verbose >= 3:
            self.logger.setLevel(logging.DEBUG)
        elif self.verbose == 2:
            self.logger.setLevel(logging.INFO)
        elif self.verbose == 1:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)

    def clean_aam(
        self, list_aam: Iterable[str], remove_radical: bool = True
    ) -> List[str]:
        """
        Remove atom-atom mappings, optionally clean radicals, deduplicate while preserving order.
        """
        seen = set()
        unique = []
        for smi in list_aam:
            try:
                no_aam = self.std.fit(smi, remove_aam=True)
                if remove_radical:
                    no_aam = clean_radical_rsmi(no_aam)
                if no_aam and no_aam not in seen:
                    seen.add(no_aam)
                    unique.append(no_aam)
            except Exception as exc:
                self.logger.debug(
                    "Failed to clean AAM for %r: %s",
                    smi,
                    exc,
                    exc_info=self.verbose >= 3,
                )
        return unique

    def _rxn_has_both_sides(self, rxn: Any) -> bool:
        if not isinstance(rxn, str):
            return False
        if ">>" in rxn:
            left, right = rxn.split(">>", 1)
            return bool(left.strip()) and bool(right.strip())
        parts = rxn.split(">")
        if len(parts) == 3:
            reactants, _, products = parts
            return bool(reactants.strip()) and bool(products.strip())
        return False

    def _process_single(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker: standardize reaction and clean fw/bw lists based on configured keys.
        """
        result = dict(value)  # shallow copy

        # Standardize reaction
        raw_rxn = value.get(self.reaction_key)
        if raw_rxn is None:
            self.logger.warning(
                "Missing reaction key '%s' in value: %r", self.reaction_key, value
            )
            result["std_rxn"] = None
        else:
            try:
                result["std_rxn"] = self.std.fit(raw_rxn)
            except Exception as exc:
                self.logger.warning("Standardization failed for %r: %s", raw_rxn, exc)
                result["std_rxn"] = None

        # Clean fw list
        try:
            raw_fw = value.get(self.fw_key, [])
            cleaned_fw = self.clean_aam(raw_fw)
            result[self.fw_key] = cleaned_fw
        except Exception as exc:
            self.logger.debug(
                "Failed cleaning fw (%s) for %r: %s",
                self.fw_key,
                value,
                exc,
                exc_info=self.verbose >= 3,
            )
            result[self.fw_key] = []

        # Clean bw list
        try:
            raw_bw = value.get(self.bw_key, [])
            cleaned_bw = self.clean_aam(raw_bw)
            result[self.bw_key] = cleaned_bw
        except Exception as exc:
            self.logger.debug(
                "Failed cleaning bw (%s) for %r: %s",
                self.bw_key,
                value,
                exc,
                exc_info=self.verbose >= 3,
            )
            result[self.bw_key] = []

        return result

    def _run_serial(
        self, seq: List[Dict[str, Any]], progress: bool
    ) -> List[Dict[str, Any]]:
        results = []
        iterator = seq
        if progress:
            if tqdm:
                iterator = tqdm(seq, desc="PostSyn", unit="item")
            else:
                self.logger.info("tqdm not available; running without progress bar.")
        for v in iterator:
            results.append(self._process_single(v))
        return results

    def _run_parallel(
        self, seq: List[Dict[str, Any]], progress: bool
    ) -> List[Dict[str, Any]]:
        if progress and tqdm:
            try:
                with _TqdmJoblib(tqdm(total=len(seq), desc="PostSyn", unit="item")):
                    return Parallel(
                        n_jobs=self.n_jobs, backend="loky", verbose=self.verbose
                    )(delayed(self._process_single)(v) for v in seq)
            except Exception as e:
                self.logger.warning(
                    "Progress-wrapped parallel execution failed (%s); falling back to normal Parallel.",
                    e,
                )
        elif progress and not tqdm:
            self.logger.info(
                "tqdm not installed; running parallel without progress bar."
            )
        return Parallel(n_jobs=self.n_jobs, backend="loky", verbose=self.verbose)(
            delayed(self._process_single)(v) for v in seq
        )

    def _filter_fw_bw(self, results: List[Dict[str, Any]]) -> None:
        check = self._rxn_has_both_sides
        for value in results:
            fw = value.get(self.fw_key, [])
            bw = value.get(self.bw_key, [])
            filtered_fw = [r for r in fw if check(r)]
            filtered_bw = [r for r in bw if check(r)]
            if self.verbose >= 3:
                dropped_fw = len(fw) - len(filtered_fw)
                dropped_bw = len(bw) - len(filtered_bw)
                if dropped_fw:
                    self.logger.debug(
                        "Dropped %d incomplete fw entries for record %r", dropped_fw, fw
                    )
                if dropped_bw:
                    self.logger.debug(
                        "Dropped %d incomplete bw entries for record %r", dropped_bw, bw
                    )
            value[self.fw_key] = filtered_fw
            value[self.bw_key] = filtered_bw

    def process(
        self,
        data: Iterable[Dict[str, Any]],
        *,
        progress: bool = False,
        prefilter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        filter_incomplete_rxn: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process reaction entries.

        :param data: iterable of dicts.
        :param progress: show progress bar if True.
        :param prefilter: predicate to pre-filter entries.
        :param filter_incomplete_rxn: if True, drop incomplete SMILES inside fw/bw.
        :return: processed list with standardized reaction and cleaned fw/bw under their original keys.
        """
        seq = list(data)
        if prefilter:
            seq = [v for v in seq if prefilter(v)]

        if self.n_jobs == 1:
            results = self._run_serial(seq, progress)
        else:
            results = self._run_parallel(seq, progress)

        if filter_incomplete_rxn:
            self._filter_fw_bw(results)

        return results
