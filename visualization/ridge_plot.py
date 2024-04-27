
# import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def combine_columns(row):
    return f"{row['year']}-{row['month']}"

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

flights = sns.load_dataset('flights')
# flights['year-month'] = flights.apply(combine_columns, axis=1)
# flights['year-month'] = flights.apply(lambda row: f"{row['year']}-{row['month']}", axis=1)
print(flights.head())
print(flights['year'].unique())

pal = sns.cubehelix_palette(len(flights["year"].unique()), start=1.4, rot=-.25, light=.7, dark=.4)

# flights = flights[flights['year'] < 1950]
g = sns.FacetGrid(flights, row='year', hue='year', aspect=20, height=.5, palette=pal)

g.map(sns.kdeplot, "passengers", bw_adjust=.6, cut=5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "passengers", bw_adjust=.6, cut=5, clip_on=False, color="w", lw=2)
g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .1, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "year")
g.fig.subplots_adjust(hspace=-.7)
g.set(yticks=[], xlabel="", ylabel="", xlim=(None, 680), title="")
g.despine(bottom=True, left=True)
plt.show()
