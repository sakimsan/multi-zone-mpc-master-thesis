import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
from bes_rules.plotting import EBCColors


def plot_tennet():
    """
    https://www.techem.com/content/dam/techem/downloads/vkw-studie/Techem-Verbrauchskennwerte-Studie-VKW%202021.pdf.coredownload.inline.pdf
    :return:
    """
    data = np.array([[0, 10.017573260497187],
            [0.16637007545946947, 12.189546780241649],
            [0.4292097588074625, 15.217230683764633],
            [0.6017795472431834, 17.38952826258043],
            [0.7546983935928894, 19.496236285357153],
            [0.9312902180454611, 21.272987361696213],
            [1.103874820610157, 23.379371325401596],
            [1.319738900976807, 25.41912874403962],
            [1.5042562844312766, 26.932095736308497],
            [1.69671404101662, 28.115365029396784],
            [1.8342854497476972, 28.508578306559876],
            [1.9404360909217173, 28.70456923290588],
            [2.058415814082682, 28.768538493588256],
            [2.1528707004305363, 28.50332854960418],
            [2.2748872737373276, 28.105837692699406],
            [2.385134021573075, 27.576714041016615],
            [2.519024119253739, 26.84946067311698],
            [2.6530179158372302, 25.660811999444466],
            [2.7909448636637197, 24.472098513957683],
            [2.9210499513911397, 23.085773806768202],
            [3.023548909772696, 22.029470857830653],
            [3.1103300773112363, 20.907513541039762],
            [3.2246877459376884, 19.587361696217766],
            [3.3351418915790942, 18.13544743298921],
            [3.457365862691543, 16.815165964538675],
            [3.5875746493217906, 14.967445951576309],
            [3.7097986204342397, 13.647164483125778],
            [3.85174760427758, 12.062904495162257],
            [4.005510855978891, 10.412536456645526]
            ])
    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 0.75, 15 / 2.54 * 2 / 3],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )
    plt.plot(data[:, 1], data[:, 0], color=EBCColors.red)
    plt.ylabel("$SCOP_\mathrm{Mess}$ [-]")
    plt.ylim([1.5, 4])
    plt.yticks([1, 2, 3, 4])

    plt.xlabel("Häufigkeit in %")
    plt.gcf().tight_layout()
    plt.savefig(f"motivation_techem.svg")
    plt.show()


def plot_motivation(n_rows):
    """
    1. Entwicklung THG Emissions Gebäudesektor:
    https://www.dena.de/fileadmin/dena/Dokumente/Landingpages/Leitstudie_II/Gutachten/211005_DLS_Gutachten_ITG_FIW_final.pdf
    2. WP Absatzzahlen:
    https://www.waermepumpe.de/fileadmin/user_upload/waermepumpe/05_Presse/01_Pressemitteilungen/BWP_Branchenstudie_2023_DRUCK.pdf
    3. Fachkräfteprognose:
    4. Plakativ geteilt
    """
    w = {
        1: 1.4,
        2: 2.5,
        4: 3.8
    }.get(n_rows, n_rows)
    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 0.8, 15 / 2.54 * w / 3],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )

    data_thg = {
        2020: 288,
        2025: 187,
        2030: 131,
        2035: 79,
        2040: 28,
        #2045: 5
    }
    data_wp = {
        2020: 119.512,
        2025: 565.854,
        2030: 968.293,
        2035: 914.634,
        2040: 873.170,
        #2045: 785.366/1000,
    }
    data_fachkraefte = {
        2020: 29.12,
        2025: 27.84,
        2030: 26.4,
        2035: 25.3,
        2040: 24.3,
        #2045: 0
    }
    y_pos = np.arange(len(data_thg))

    fig, ax = plt.subplots(n_rows, 1, sharex=True)
    # Create bars
    if n_rows == 1:
        ax = [ax]
    barlist = ax[0].bar(y_pos, list(data_thg.values()), color=EBCColors.light_red, width=0.4)
    barlist[0].set_color(EBCColors.red)
    ax[0].set_ylabel("in Mio. t/a")
    ax[0].set_title("THG-Emissionen [2]")
    if n_rows > 1:
        barlist = ax[1].bar(y_pos, list(data_wp.values()), color=EBCColors.light_red, width=0.4)
        barlist[0].set_color(EBCColors.red)
        ax[1].set_yticks([0, 500, 1000])
        ax[1].set_ylabel("in Tsd.")
        ax[1].set_title("Verkaufszahlen WP [3]")
    if n_rows > 2:
        barlist = ax[2].bar(y_pos, list(data_fachkraefte.values()), color=EBCColors.light_red, width=0.4)
        barlist[0].set_color(EBCColors.red)
        ax[2].set_ylabel("in Mio.")
        ax[2].set_title("Fachkräftebasis [7]")
        ax[2].set_ylim([20, 30])

    ax[n_rows - 1].set_xticks(y_pos, data_thg.keys(), rotation=45)
    ax[n_rows - 1].set_xlabel("Jahre")
    import matplotlib.patches as mpatches

    #ax[0].legend(handles=[
    #    mpatches.Patch(color=EBCColors.red, label='Daten'),
    #    mpatches.Patch(color=EBCColors.light_red, label='Prognose')
    #], loc="lower left", bbox_to_anchor=(0, 1), ncol=2)

    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"motivation_slide_{n_rows}.svg")
    plt.show()


def plot_air_source_bars():
    """
    Data from:
    :return:
    """
    data = [2.9265536723163836,
            1.694915254237288,
            1.909604519774011,
            2.406779661016949,
            2.666666666666667,
            3.7288135593220333,
            3.4915254237288136,
            2.4858757062146895,
            2.677966101694915,
            2.3954802259887003,
            2.0564971751412426]
    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 0.75, 15 / 2.54 * 2 / 3],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )

    y_pos = np.arange(len(data))

    fig, ax = plt.subplots(1, 1, sharex=True)
    # Create bars
    ax.bar(y_pos, data, color=EBCColors.light_red)

    ax.set_xticks(y_pos, range(1, len(data) + 1))
    ax.set_xlabel("Wärmepumpe")
    ax.set_ylabel("$SCOP_\mathrm{Mess}$ [-]")
    ax.set_ylim([1, 4])
    ax.set_yticks([1, 2, 3, 4])

    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"motivation_field_data_air_source.svg")
    plt.show()


def plot_air_source_data_from_uk(only_phase_1=False):
    """
    Data from: https://www.energysavingtrust.org.uk/sites/default/files/reports/TheHeatisOnweb(1).pdf
    :return:
    """
    data_phase_1 = {1.2: 1, 1.4: 1, 1.6: 7, 1.8: 4, 2: 4, 2.2: 5}
    data_phase_2 = {2.0: 3, 2.2: 5, 2.4: 3, 2.6: 1, 2.8: 1, 3.0: 1, 3.6: 1}

    print("Mean Phase 1", sum([k * v for k, v in data_phase_1.items()]) / sum(data_phase_1.values()), sum(data_phase_1.values()))
    print("Mean Phase 1", sum([k * v for k, v in data_phase_2.items()]) / sum(data_phase_2.values()), sum(data_phase_2.values()))

    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 1.5, 15 / 2.54 * 1.5 / 3],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )
    data_points = [round(v, 1) for v in np.arange(1.2, 3.6, 0.2)]
    y_pos = np.arange(len(data_points))

    data_phase_1 = [data_phase_1.get(p, 0) for p in data_points]
    data_phase_2 = [data_phase_2.get(p, 0) for p in data_points]
    fig, ax = plt.subplots(1, 1, sharey=True)
    # Create bars
    ax.bar(y_pos - 0.2, data_phase_1,
           color=EBCColors.light_red, align="center", label="Phase 1", width=0.4)
    if not only_phase_1:
        ax.bar(y_pos + 0.2, data_phase_2,
               color=EBCColors.red, align="center", label="Phase 2", width=0.4)
    else:
        ax.bar(y_pos + 0.2, data_phase_2,
               color="white", align="center", label="", width=0.4)
    ax.set_xticks(y_pos, [str(p).replace(".", ",") for p in data_points])
    ax.set_ylabel("Anzahl\nWärmepumpen [6]")
    ax.set_xlabel("$SCOP_\mathrm{Mess}$")
    ax.legend(ncol=2)
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"motivation_field_data_air_uk{'_p1' if only_phase_1 else ''}.svg")
    plt.show()


def carnot_efficiency_Kozarcanin_et_al():
    def COP(T_air, T_sink):
        dT = T_sink - T_air
        if T_air <= 5:
            return 0.0012 * dT ** 2 - 0.1702 * dT + 7.855
        else:
            return 0.0019 * dT ** 2 - 0.2258 * dT + 9.073
    def carnotCOP(T_air, T_sink):
        return (273.15 + T_sink)/(T_sink - T_air)
    t_sink = 55
    t_air = np.arange(-20, 20)
    qg = [100 * COP(T_air=t, T_sink=t_sink) / carnotCOP(T_air=t, T_sink=t_sink) for t in t_air]
    plt.plot(t_air, qg)
    plt.xlabel("Außenlufttemperatur in °C")
    plt.ylabel("Gütegrad in %")
    plt.show()


def generate_word_cloud():
    words_1 = {
        "Rule mining": 2,
        "Knowledge discovery": 4,
        "Interpretable rule discovery": 1,
        "Knowledge mining": 1,
        "Knowledge extraction": 2,
        "Innovization": 1,
        "Data mining": 2,
        "Rule extraction": 3,
        "Derivation of design rules": 1,
    }
    words_2 = {
        "Design rules": 4,
        "Planning rules": 2,
        "Planning principles": 2,
        "Design principles": 5,
        "Decision rules": 1,
        "Decision principles": 1,
        #"heat pump": 5
    }
    words_3 = {
        "heat pump": 5,
        "photovoltaic": 2,
        "gas boiler": 2,
        "residential": 4,
        "building energy system": 4
    }

    from wordcloud import WordCloud

    # Generate a word cloud image
    wordcloud = WordCloud(
        background_color='white',
        max_font_size=50,
        min_font_size=10,
        collocations=False,
    ).generate_from_frequencies(words_1)

    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("wordcloud_rule_buzzwords.svg")
    wordcloud = WordCloud(
        background_color='white',
        max_font_size=50,
        min_font_size=10,
        collocations=False,
    ).generate_from_frequencies(words_2)

    # Display the generated image:
    # the matplotlib way:
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("wordcloud_design_words.svg")
    wordcloud = WordCloud(
        background_color='white',
        max_font_size=50,
        min_font_size=10,
        collocations=False,
    ).generate_from_frequencies(words_3)

    # Display the generated image:
    # the matplotlib way:
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("wordcloud_must_have_words.svg")
    plt.show()


def plot_tBiv_rules():
    plt.rcParams.update(
        {"figure.figsize": [13 / 2.54 * 1.2, 15 / 2.54 * 0.5],
         "font.size": 16,
         "figure.dpi": 250,
         "font.family": "Arial"
         }
    )
    def get_t_biv_equal(T_NA, beta, vering=False):
        T_R = 293.15
        dQ = (5.26 - 2.89) / 15
        Q_flow_nominal = 11
        UA = Q_flow_nominal / (T_R - T_NA)
        if vering:
            beta = beta - dQ / Q_flow_nominal * (271.15 - T_NA)
        return (T_R * (1 - beta) + T_NA * (dQ / UA + beta)) / (dQ / UA + 1)
    t_na = np.linspace(-16, -7, 100)

    fig, ax = plt.subplots()
    ax.axhline(-2, label="[77]", color=EBCColors.red, linewidth=3)
    #ax.axhline(2, label="Hybrid [13]", color=EBCColors.light_red)
    #ax.plot([-16] * 2, [-8, -4], label="EE.org [14]", color=EBCColors.blue)
    ax.plot([-16] * 2, [-7, -4], label="[21,78]", color=EBCColors.blue, linewidth=3)
    ax.plot([-12-0.05] * 2, [-6, -3], color=EBCColors.blue, linewidth=3)
    ax.plot([-10-0.05] * 2, [-5, -2], color=EBCColors.blue, linewidth=3)
    #ax.plot(t_na, get_t_biv_equal(T_NA=t_na + 273.15, beta=0.8) - 273.15, label="[Z]", color=EBCColors.dark_grey, linewidth=3)
    #ax.plot(t_na, get_t_biv_equal(T_NA=t_na + 273.15, beta=1) - 273.15, label="[8]", color=EBCColors.dark_red)
    for i, t_na_v in enumerate([-12, -10, -9]):
        _max_beta = get_t_biv_equal(T_NA=t_na_v + 273.15, beta=0.75, vering=True) - 273.15
        _min_beta = get_t_biv_equal(T_NA=t_na_v + 273.15, beta=1.2, vering=True) - 273.15
        if i == 0:
            ax.plot([t_na_v+0.05] * 2, [_min_beta, _max_beta], label="[76]", color=EBCColors.grey, linewidth=3)
        else:
            ax.plot([t_na_v+0.05] * 2, [_min_beta, _max_beta], color=EBCColors.grey, linewidth=3)
    ax.scatter([-9], get_t_biv_equal(T_NA=-9 + 273.15, beta=0.74) - 273.15, label="[66]", color="black", marker="x", s=100)
    ax.grid()
    ax.legend(ncol=1, bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_xlabel("$T_\mathrm{NA}$ in °C")
    ax.set_ylabel("$T_\mathrm{Biv}$ in °C")
    fig.tight_layout()
    fig.savefig("TBivSuggestions.svg")
    plt.show()

if __name__ == '__main__':
    #[plot_motivation(i) for i in range(1, 4)]
    #plot_motivation(with_eff=False)
    #plot_motivation_only_thg()
    #plot_tennet()
    #plot_air_source_bars()
    #plot_air_source_data_from_uk(only_phase_1=True)
    plot_air_source_data_from_uk(only_phase_1=False)

    #plot_tBiv_rules()
