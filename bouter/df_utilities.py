import numpy as np


def tail_column_names(n_segments, i_fish=None):
    if i_fish is not None:
        return [
            "f{:d}_theta_{:02d}".format(i_fish, i) for i in range(n_segments)
        ]
    else:
        return ["theta_{:02d}".format(i) for i in range(n_segments)]


def get_n_segments(df, prefix=True):
    if prefix:

        def _tail_part(s):
            ps = s.split("_")
            if len(ps) == 3:
                return ps[2]
            else:
                return 0

    else:

        def _tail_part(s):
            ps = s.split("_")
            if len(ps) == 2:
                return ps[1]
            else:
                return 0

    tpfn = np.vectorize(_tail_part, otypes=[int])
    return np.max(tpfn(df.columns.values)) + 1


def get_n_segments_embedded(df):
    n_segs = 0
    for col in df.columns:
        if col.startswith("theta_"):
            n_segs = max(n_segs, int(col.split("_")[1]))
    return n_segs + 1
