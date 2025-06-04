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
    "row-row": "Row-\nNN",
    "col-col": "Col-\nNN",
    "dr": "DR-\nNN",
    "ts": "TS-\nNN",
    "usvt": "USVT",
    "softimpute": "Soft\nImpute",
    "nadaraya": "NW",
    "star": "AW-\nNN",
    "sc": "SC",
    "auto": "Auto-\nNN",
}

METHOD_ALIASES_SINGLE_LINE = {
    "row-row": "RowNN",
    "col-col": "ColNN",
    "dr": "DRNN",
    "ts": "TSNN",
    "usvt": "USVT",
    "softimpute": "SI",
    "nadaraya": "NW",
    "star": "AWNN",
    "sc": "SC",
    "auto": "AutoNN",
}

METHOD_LINE_STYLES = {
    "row-row": "--",
    "col-col": "--",
    "dr": "dotted",
    "ts": "dotted",
    "usvt": "-",
    "softimpute": "-",
    "nadaraya": "-",
    "star": "dashdot",
    "sc": "-",
    "auto": "-",
}

NEURIPS_TEXTWIDTH = 5.5

COLORS = {
    "row-row": "red",
    "col-col": "grey",
    "dr": "grey",
    "ts": "grey",
    "usvt": "lightgrey",
    "softimpute": "lightgrey",
    "star": "grey",
    "sc": "blue",
    "auto": "grey",
}

DATA_TYPE_ALIASES = {
    "kernel_mmd": "MMD",
    "wasserstein_samples": "W$_2$",
}
