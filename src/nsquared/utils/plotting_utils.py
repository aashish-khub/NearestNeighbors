# Enable LaTeX for text rendering
import matplotlib as mpl

mpl.rcParams["text.latex.preamble"] = r"\usepackage{amssymb} \usepackage{amsmath}"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


# Single column figures
TICK_FONT_SIZE = 6
MINOR_TICK_FONT_SIZE = 3
LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 7
MARKER_ALPHA = 1
MARKER_SIZE = 3
LINE_ALPHA = 0.75
OUTWARD = 4

METHOD_ALIASES = {
    "kernel": "Kernel-\nNN",
    "wasserstein_samples": "W$_2$-\nNN",
    "auto": "Auto-\nNN",
    "row-row": "Row-\nNN",
    "col-col": "Col-\nNN",
    "dr": "DR-\nNN",
    "ts": "TS-\nNN",
    "usvt": "USVT",
    "softimpute": "Soft\nImpute",
    "nadaraya": "NW",
    "aw": "AW-\nNN",
    "sc": "SC",
}

METHOD_ALIASES_SINGLE_LINE = {
    "auto": "AutoNN",
    "row-row": "RowNN",
    "col-col": "ColNN",
    "dr": "DRNN",
    "ts": "TSNN",
    "usvt": "USVT",
    "softimpute": "SI",
    "nadaraya": "NW",
    "aw": "AWNN",
    "sc": "SC",
    "kernel": "KernelNN",
    "wasserstein_samples": "W$_2$NN",
}

METHOD_LINE_STYLES = {
    "row-row": "--",
    "col-col": "--",
    "dr": "dotted",
    "ts": "dotted",
    "usvt": "-",
    "softimpute": "-",
    "nadaraya": "-",
    "aw": "dashdot",
    "sc": "-",
    "auto": "-",
}

NEURIPS_TEXTWIDTH = 5.5

COLORS = {
    "row-row": "red",
    "auto": "grey",
    "kernel": "dimgrey",
    "wasserstein_samples": "dimgrey",
    # "row-row": "grey",
    "col-col": "grey",
    "dr": "grey",
    "ts": "grey",
    "usvt": "lightgrey",
    "softimpute": "lightgrey",
    "aw": "grey",
    "sc": "blue",
}

DATA_TYPE_ALIASES = {
    "kernel_mmd": "MMD",
    "wasserstein_samples": "W$_2$",
}
