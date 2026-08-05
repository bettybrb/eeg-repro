import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def apply_compatibility_patches():
    if not hasattr(np, "str"):
        np.str = str
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "complex"):
        np.complex = complex

    original_clip = np.clip

    def safe_clip_for_old_mne_gdf(a, a_min=None, a_max=None, out=None, **kwargs):
        if a_min is None:
            a_min = kwargs.pop("minimum", None)
        if a_max is None:
            a_max = kwargs.pop("maximum", None)

        if (
            out is not None
            and hasattr(out, "dtype")
            and out.dtype.kind in ("u", "i")
            and np.isscalar(a_max)
            and np.isinf(a_max)
        ):
            a_max = np.iinfo(out.dtype).max

        return original_clip(a, a_min, a_max, out=out, **kwargs)

    np.clip = safe_clip_for_old_mne_gdf

    if not hasattr(pd.DataFrame, "append"):
        def dataframe_append_compat(self, other, ignore_index=False, **kwargs):
            if isinstance(other, dict):
                other = pd.DataFrame([other])
            return pd.concat([self, other], ignore_index=ignore_index)

        pd.DataFrame.append = dataframe_append_compat

    if not hasattr(pd.Series, "iteritems"):
        pd.Series.iteritems = pd.Series.items


class UsefulProgressHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()

        if (
            msg.startswith("START")
            or msg.startswith("DONE")
            or msg.startswith("FAILED")
            or msg.startswith("SUMMARY")
            or msg.startswith("SKIP")
        ):
            print(msg, flush=True)

        elif msg.startswith("Epoch "):
            print(msg, flush=True)


def setup_logging():
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.INFO)
    root.addHandler(UsefulProgressHandler())
