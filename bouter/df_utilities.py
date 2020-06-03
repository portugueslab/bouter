import numpy as np


def tail_column_names(n_segments, i_fish=None):
    if i_fish is not None:
        return ["f{:d}_theta_{:02d}".format(i_fish, i) for i in range(n_segments)]
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


def get_n_fish(df):
    def _fish_part(s):
        ps = s.split("_")
        if len(ps) == 3:
            return ps[0][1:]
        else:
            return 0

    tpfn = np.vectorize(_fish_part, otypes=[int])
    return np.max(tpfn(df.columns.values)) + 1


def _fish_column_names(i_fish, n_segments):
    return [
        "f{:d}_x".format(i_fish),
        "f{:d}_vx".format(i_fish),
        "f{:d}_y".format(i_fish),
        "f{:d}_vy".format(i_fish),
        "f{:d}_theta".format(i_fish),
        "f{:d}_vtheta".format(i_fish),
    ] + ["f{:d}_theta_{:02d}".format(i_fish, i) for i in range(n_segments)]


def _fish_renames(i_fish, n_segments):
    return dict(
        {
            "f{:d}_x".format(i_fish): "x",
            "f{:d}_vx".format(i_fish): "vx",
            "f{:d}_y".format(i_fish): "y",
            "f{:d}_vy".format(i_fish): "vy",
            "f{:d}_theta".format(i_fish): "theta",
            "f{:d}_vtheta".format(i_fish): "vtheta",
        },
        **{
            "f{:d}_theta_{:02d}".format(i_fish, i): "theta_{:02d}".format(i)
            for i in range(n_segments)
        }
    )


def _rename_fish(df, i_fish, n_segments):
    return df.filter(["t"] + _fish_column_names(i_fish, n_segments)).rename(
        columns=_fish_renames(i_fish, n_segments)
    )
