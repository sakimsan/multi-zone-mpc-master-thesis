import seaborn as sns
from bes_rules.plotting import EBCColors


def plot_box(df, orient, axes, color_palette = None):
    if color_palette is None:
        sns.boxplot(data=df, ax=axes, orient=orient, color=EBCColors.light_grey,
                    showfliers=False, palette=color_palette)
    else:
        sns.boxplot(data=df, ax=axes, orient=orient,
                    showfliers=False, palette=color_palette)
    for i, col in enumerate(df.columns, 0):
        if orient == "v":
            axes.scatter([i, i], [df[col].min(), df[col].max()], marker='d', facecolor='black')
        else:
            axes.scatter([df[col].min(), df[col].max()], [i, i], marker='d', facecolor='black')
