from bes_rules import RESULTS_FOLDER
from studies_ssr.sfh_PIctrl_monovalent_spawn.plotting.plot_PI_discomfort_CISBAT import load_discomfort_dict_json, plot_discomfort_values
from studies_ssr.sfh_mpc_hom_monovalent_spawn.postprocessing.save_mpc_discomfort_results import get_heated_zone_names

if __name__ == "__main__":
    # load the saved dict with discomfort results
    discomfort_dict = load_discomfort_dict_json(filename_path=RESULTS_FOLDER.joinpath("SFH_MPCRom_monovalent_spawn", "discomfort_data.json"))
    # load zone names
    zone_names = get_heated_zone_names(with_sum=True, with_TZoneAreaWeighted=True)
    # create heat map with discomfort values of zone name
    for zone_name in zone_names:
        plot_discomfort_values(
            data=discomfort_dict,
            zone=zone_name,
            show_values=True,
            show_vdi_line=False,
            save_pdf=False,
            save_png=True,
            save_svg=False
        )