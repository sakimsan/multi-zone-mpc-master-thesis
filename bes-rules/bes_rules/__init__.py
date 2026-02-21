from pathlib import Path
import json
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


REPO_ROOT = Path(__file__).absolute().parents[1]
PC_SPECIFIC_SETTING_PATH = REPO_ROOT.joinpath("pc_specific_settings.json")
if os.path.exists(PC_SPECIFIC_SETTING_PATH):
    with open(PC_SPECIFIC_SETTING_PATH, "r") as file:
        PC_SPECIFIC_SETTINGS = json.load(file)
    STARTUP_BESMOD_MOS = Path(PC_SPECIFIC_SETTINGS["STARTUP_BESMOD_MOS"])
    N_CPU = PC_SPECIFIC_SETTINGS["N_CPU"]
    RESULTS_FOLDER = Path(PC_SPECIFIC_SETTINGS["RESULTS_FOLDER"])
    REF_PROP_PATH = Path(PC_SPECIFIC_SETTINGS["REF_PROP_PATH"])
    BESGRICONOP_PACKAGE_MO = Path(PC_SPECIFIC_SETTINGS["BESGRICONOP_PACKAGE_MO"]).joinpath("Modelica", "BESGriConOp", "package.mo")
    #LATEX_FIGURES_FOLDER = Path(PC_SPECIFIC_SETTINGS["LATEX_FIGURES_FOLDER"])
else:
    STARTUP_BESMOD_MOS = Path(r"D:\04_git\BESMod\startup.mos")
    N_CPU = 9
    RESULTS_FOLDER = Path(r"D:\00_temp\01_bes_rules")
    REF_PROP_PATH = Path(r"D:\02_Paper\vclibpy\REFPROP")
    LATEX_FIGURES_FOLDER = Path(r"D:\02_Paper\diss_latex\Figures")

DATA_PATH = REPO_ROOT.joinpath("data")
BESRULES_PACKAGE_MO = REPO_ROOT.joinpath("modelica", "BESRules", "package.mo")
