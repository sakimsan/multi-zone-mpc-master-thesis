from studies.use_case_1_design.hp_influences import compare_onoff, compare_partload_optihorst

if __name__ == '__main__':
    compare_onoff.run(study_name="inverter_vs_onoff_hydSep_debug", n_cpu=12)
    compare_partload_optihorst.run(n_cpu=12)
