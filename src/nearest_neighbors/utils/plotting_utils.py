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
    "softimpute": "SoftImpute",
    "nadaraya": "NW",
    "star": "Star-\nNN",
}

METHOD_ALIASES_SINGLE_LINE = {
    "row-row": "Row-NN",
    "col-col": "Col-NN",
    "dr": "DR-NN",
    "ts": "TS-NN",
    "usvt": "USVT",
    "softimpute": "SoftImpute",
    "nadaraya": "NW",
    "star": "Star-NN",
    "sc": "SC",
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
}

NEURIPS_TEXTWIDTH = 5.5
