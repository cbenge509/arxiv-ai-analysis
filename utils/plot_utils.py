############################################################################
# IMPORTS
############################################################################

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
from bokeh.models import CustomJS
import os
import bokeh
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, Slider, TapTool, TextInput
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap, transform
from bokeh.io import output_file, show, save, output_notebook
from bokeh.plotting import figure
from bokeh.models import RadioButtonGroup, TextInput, Div, Paragraph
from bokeh.layouts import column, widgetbox, row, layout
from bokeh.layouts import column


############################################################################
# Plotting Utilities, Constants, Methods for W209 arXiv project
############################################################################

#---------------------------------------------------------------------------
## Plotting Palette
#
# Create a dict object containing U.C. Berkeley official school colors for plot palette 
# reference : https://brand.berkeley.edu/colors/
# secondary reference : https://alumni.berkeley.edu/brand/color-palette# CLass Initialization
#---------------------------------------------------------------------------

berkeley_palette = {'berkeley_blue'     : '#003262',
                    'california_gold'   : '#fdb515',
                    'founders_rock'     : '#3b7ea1',
                    'medalist'          : '#c4820e',
                    'bay_fog'           : '#ddd5c7',
                    'lawrence'          : '#00b0da',
                    'sather_gate'       : '#b9d3b6',
                    'pacific'           : '#46535e',
                    'soybean'           : '#859438',
                    'south_hall'        : '#6c3302',
                    'wellman_tile'      : '#D9661F',
                    'rose_garden'       : '#ee1f60',
                    'golden_gate'       : '#ed4e33',
                    'lap_lane'          : '#00a598',
                    'ion'               : '#cfdd45',
                    'stone_pine'        : '#584f29',
                    'grey'              : '#eeeeee',
                    'web_grey'          : '#888888',
                    # alum only colors
                    'metallic_gold'     : '#BC9B6A',
                    'california_purple' : '#5C3160'                   
                    }

#---------------------------------------------------------------------------
## Altair custom "Cal" theme
#---------------------------------------------------------------------------

def cal_theme():
    font = "Lato"

    return {
        "config": {
            "title": {
                "fontSize": 25,
                "font": font,
                "anchor": "middle",
                "align":"center",
                "color": berkeley_palette['berkeley_blue'],
                "subtitleFontSize": 15,
                "subtitleFont": font,
                "subtitleAcchor": "middle",
                "subtitleAlign": "center",
                "subtitleColor": berkeley_palette['pacific']
            },
            "axisX": {
                "labelFont": font,
                "titleFont": font,
                "titleFontSize": 15,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            },
            "axisY": {
                "labelFont": font,
                "titleFont": font,
                "titleFontSize": 12,
                "titleColor": berkeley_palette['pacific']
            },
            "headerRow": {
                "labelFont": font,
                "titleFont": font,
                "titleFontSize": 15,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            },
            "legend": {
                "labelFont": font,
                "labelFontSize": 10,
                "labelColor": berkeley_palette['stone_pine'],
                "symbolType": "circle",
                "symbolSize": 120,
                "titleFont": font,
                "titleFontSize": 12,
                "titleColor": berkeley_palette['pacific']
            }
        }
    }

alt.themes.register("my_cal_theme", cal_theme)
alt.themes.enable("my_cal_theme")

#---------------------------------------------------------------------------
## Bayesian review of paper sub-categories
#---------------------------------------------------------------------------

def bayes_me_please(df, priors_color = berkeley_palette['medalist'], intersects_color = berkeley_palette['pacific'], joint_probs_color = berkeley_palette['california_purple']):

    def go_bayes_or_go_home(P_A, P_B, P_B_given_A):
        return (P_B_given_A * P_A) / (P_B)

    tot, df_joint_prob = df.shape[0], {}

    df_joint_prob["P(cs.AI)"] = df[(df.category_cs_AI == True)].shape[0] / tot
    df_joint_prob["P(cs.LG)"] = df[(df.category_cs_LG == True)].shape[0] / tot
    df_joint_prob["P(stat.ML)"] = df[(df.category_stat_ML == True)].shape[0] / tot

    df_joint_prob["P(cs.AI,cs.LG)"] = df[(df.category_cs_AI == True) & (df.category_cs_LG == True)].shape[0] / tot
    df_joint_prob["P(cs.AI,stat.ML)"] = df[(df.category_cs_AI == True) & (df.category_stat_ML == True)].shape[0] / tot
    df_joint_prob["P(cs.LG,stat.ML)"] = df[(df.category_cs_LG == True) & (df.category_stat_ML == True)].shape[0] / tot
    df_joint_prob["P(cs.AI,cs.LG,stat.ML)"] = df[(df.category_cs_AI == True) & (df.category_stat_ML == True) & (df.category_cs_LG == True)].shape[0] / tot

    df_joint_prob["P(cs.AI|cs.LG)"] = df[(df.category_cs_AI == True) & (df.category_cs_LG == True)].shape[0] / df[(df.category_cs_LG)].shape[0]
    df_joint_prob["P(cs.LG|cs.AI)"] = go_bayes_or_go_home(P_A = df_joint_prob["P(cs.LG)"], P_B = df_joint_prob["P(cs.AI)"], P_B_given_A = df_joint_prob["P(cs.AI|cs.LG)"])

    df_joint_prob["P(cs.AI|stat.ML)"] = df[(df.category_cs_AI == True) & (df.category_stat_ML == True)].shape[0] / df[(df.category_stat_ML)].shape[0]
    df_joint_prob["P(stat.ML|cs.AI)"] = go_bayes_or_go_home(P_A = df_joint_prob["P(stat.ML)"], P_B = df_joint_prob["P(cs.AI)"], P_B_given_A = df_joint_prob["P(cs.AI|stat.ML)"])

    df_joint_prob["P(cs.LG|stat.ML)"] = df[(df.category_cs_LG == True) & (df.category_stat_ML == True)].shape[0] / df[(df.category_stat_ML)].shape[0]
    df_joint_prob["P(stat.ML|cs.LG)"] = go_bayes_or_go_home(P_A = df_joint_prob["P(stat.ML)"], P_B = df_joint_prob["P(cs.LG)"], P_B_given_A = df_joint_prob["P(cs.LG|stat.ML)"])

    df_joint_prob["P(cs.AI|stat.ML,cs.LG)"] = df[(df.category_cs_AI == True) & (df.category_stat_ML == True) & (df.category_cs_LG == True)].shape[0] / df[(df.category_cs_LG == True) & (df.category_stat_ML == True)].shape[0]
    df_joint_prob["P(cs.LG|stat.ML,cs.AI)"] = df[(df.category_cs_AI == True) & (df.category_stat_ML == True) & (df.category_cs_LG == True)].shape[0] / df[(df.category_cs_AI == True) & (df.category_stat_ML == True)].shape[0]
    df_joint_prob["P(stat.ML|cs.LG,cs.AI)"] = df[(df.category_cs_AI == True) & (df.category_stat_ML == True) & (df.category_cs_LG == True)].shape[0] / df[(df.category_cs_LG == True) & (df.category_cs_AI == True)].shape[0]

    df_joint_prob = pd.DataFrame(df_joint_prob, index = [0]).T
    df_joint_prob[0] = df_joint_prob[0] * 100.
    df_joint_prob.columns = ['probability']

    # Test series

    priors = df_joint_prob.loc[["P(cs.AI)","P(cs.LG)","P(stat.ML)"]].probability
    intersects = df_joint_prob.loc[["P(cs.AI,cs.LG)", "P(cs.AI,stat.ML)", "P(cs.LG,stat.ML)", "P(cs.AI,cs.LG,stat.ML)"]].probability
    joint_probs = df_joint_prob.loc[["P(cs.AI|cs.LG)", "P(cs.LG|cs.AI)", "P(cs.AI|stat.ML)", "P(stat.ML|cs.AI)", "P(cs.LG|stat.ML)", "P(stat.ML|cs.LG)", 
        "P(cs.AI|stat.ML,cs.LG)", "P(cs.LG|stat.ML,cs.AI)", "P(stat.ML|cs.LG,cs.AI)"]].probability

    head = """
    <h1 align = "left">Time to get Bayesian!</h1>
    <table>
        <thead>
            <th></th>
            <th>Probabilities</th>
        </thead>
        </tbody>
    """

    segments = zip(['Priors','Intersects','Joint Probs'], [priors, intersects, joint_probs], 
        [priors_color, intersects_color, joint_probs_color])
    for l, s, a in segments:
        row = f"<tr><th>{l}</th>"
        sc = s.copy()
        sc.name = ""
        #row += "<td>{}</td>".format(sc.to_frame().style.bar(align = a, color=['#d65f5f', '#5fba7d'], width = 100).render()) #testn['width']
        cm = sns.light_palette(a, as_cmap = True)
        row += "<td>{}</td>".format(sc.to_frame().style.background_gradient(cmap = cm).render())
        row += '</tr>'
        head += row

    head+= """
    </tbody>
    </table>"""

    return head


#---------------------------------------------------------------------------
## Pareto Plot
#---------------------------------------------------------------------------

def pareto_plot(df, tot, x, y, x_label = None, y_label = None, title = None, show_pct_y = False, pct_format='{0:.0%}'):

    df = df.sort_values(y, ascending = False)
    x = df[x].values
    y = df[y].values

    fig = plt.figure(figsize=(10, 6), dpi = 100)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.bar(x = x, height = y, width = 0.9, align = 'center', edgecolor = berkeley_palette['berkeley_blue'],
        color = berkeley_palette['pacific'], linewidth = 1, alpha = 0.8)
    ax1.set_xticks(range(df.shape[0]))
    ax1.set_xticklabels(x, rotation = 45)
    ax1.get_yaxis().set_major_formatter(
        tck.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax1.tick_params(axis = 'y', labelsize = 8)
    ax1.tick_params(axis = 'y', labelcolor = berkeley_palette['pacific'])

    if x_label:
        ax1.set_xlabel(x_label, fontsize = 15, horizontalalignment = 'right', x = 1.0, color = berkeley_palette['berkeley_blue'])
    if y_label:
        ax1.set_ylabel(y_label, fontsize = 15, horizontalalignment = 'right', y = 1.0, color = berkeley_palette['berkeley_blue'])
    if title:
        plt.title(title, fontsize = 20, fontweight = 'bold', color = berkeley_palette['berkeley_blue'])

    weights = y / tot
    cumsum = weights.cumsum()

    ax2.plot(x, cumsum, color = berkeley_palette['south_hall'], label = 'Cumulative Distribution', alpha = 0.5)
    ax2.scatter(x, cumsum, color = berkeley_palette['california_purple'], marker = 'D', s = 5)
    ax2.set_ylabel('', color = berkeley_palette['berkeley_blue'])
    ax2.tick_params('y', colors = berkeley_palette['pacific'])
    ax2.set_ylim(0, 1.05)
    
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals], fontsize = 8)

    # hide y-labels on right side
    if not show_pct_y:
        ax2.set_yticks([])

    formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, xy = (x[i], cumsum[i] + .01), fontweight = 'bold', color = berkeley_palette['berkeley_blue'])

    # Adjust the plot spine borders to be lighter
    for ax in [ax1, ax2]:
        for p, v in zip(["top", "bottom", "right", "left"], [0.0, 0.3, 0.0, 0.3]):
            ax.spines[p].set_alpha(v)

    # Sset the Y-axis grid-lines to dim, and display the Accuracy plot.
    plt.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------------
## Venn3 Plot
#---------------------------------------------------------------------------

def plot_3venn(df, height, width, title):

    csAI = [i for i in df[(df.category_cs_AI == True)].index.values]
    csLG = [i for i in df[(df.category_cs_LG == True)].index.values]
    statML = [i for i in df[(df.category_stat_ML == True)].index.values]

    ## Generate the plot

    fig = plt.figure(figsize = (height, width), dpi = 100)
    ax = fig.add_subplot(111)

    v = venn3([set(csAI), set(csLG), set(statML)], set_labels = ('cs.AI', 'cs.LG', 'stat.ML'))
    # set the bubbles to Berkeley colors
    for id, col in zip(['001', '010', '011', '100', '101', '111', 'A', 'B', 'C'],
            [berkeley_palette['bay_fog'], berkeley_palette['california_gold'], berkeley_palette['soybean'],
            berkeley_palette['lawrence'], berkeley_palette['pacific'], berkeley_palette['berkeley_blue'],
            berkeley_palette['lawrence'], berkeley_palette['california_gold'], berkeley_palette['pacific']]):
        x = v.get_patch_by_id(id)
        x.set_alpha(0.4)
        x.set_color(col)

    # set the label colors to Berkeley blue
    for id, col in zip(['A','B','C'], [berkeley_palette['berkeley_blue']] * 3):
        v.get_label_by_id(id).set_color(col)

    ax.set_title(title, color = berkeley_palette['berkeley_blue'], fontsize = 15, fontweight = "bold")
    ax.spines["top"].set_alpha(.0)
    ax.spines["bottom"].set_alpha(.3)
    ax.spines["right"].set_alpha(.0)
    ax.spines["left"].set_alpha(.3)

    plt.show()


#---------------------------------------------------------------------------
## 3-facet scatterplot for abstract and author analysis
#---------------------------------------------------------------------------

def plot_3facet_scatter(df, fig_height, fig_width, sup_title, sup_title_color = berkeley_palette['wellman_tile']):

    fig = plt.figure(figsize = (fig_width, fig_height), dpi = 100)
    ax = fig.add_subplot(131)

    abs_data = df.groupby(pd.Grouper(key = 'created', freq = 'M')).abstract_word_count.mean().to_frame().ffill()
    #ax.scatter(list(range(len(abs_data))), abs_data, color = berkeley_palette['berkeley_blue'])
    ax.scatter(abs_data.index, abs_data, color = berkeley_palette['lawrence'], alpha = 0.5, edgecolor = berkeley_palette['berkeley_blue'])
    m, b = np.polyfit(abs_data.index.astype(int), abs_data, 1)
    ax.plot(abs_data.index, m*abs_data.index.astype(int) + b, color = berkeley_palette['pacific'], linestyle = '--', 
        alpha = 1, linewidth = 3)
    ax.set_xlabel("Publication Date", fontsize = 12, horizontalalignment = 'right', x = 1.0, color = berkeley_palette['berkeley_blue'])
    ax.set_ylabel("Mean Abstract Size (in Words)", fontsize = 12, horizontalalignment = 'right', y = 1.0, color = berkeley_palette['berkeley_blue'])
    plt.title("Abstract Size in Words\nArXiV (1993 - 2019)", fontsize = 15, fontweight = 'bold', color = berkeley_palette['berkeley_blue'])

    ax = fig.add_subplot(132)

    abs_data = df.groupby(pd.Grouper(key = 'created', freq = 'M')).abstract_sentence_count.mean().to_frame().ffill()
    #ax.scatter(list(range(len(abs_data))), abs_data, color = berkeley_palette['berkeley_blue'])
    ax.scatter(abs_data.index, abs_data, color = berkeley_palette['founders_rock'], alpha = 0.5, edgecolor = berkeley_palette['south_hall'])
    m, b = np.polyfit(abs_data.index.astype(int), abs_data, 1)
    ax.plot(abs_data.index, m*abs_data.index.astype(int) + b, color = berkeley_palette['pacific'], linestyle = '--', 
        alpha = 1, linewidth = 3)
    ax.set_xlabel("Publication Date", fontsize = 12, horizontalalignment = 'right', x = 1.0, color = berkeley_palette['berkeley_blue'])
    ax.set_ylabel("Mean Abstract Size (in Sentences)", fontsize = 12, horizontalalignment = 'right', y = 1.0, color = berkeley_palette['berkeley_blue'])
    plt.title("Abstract Size in Sentences\nArXiV (1993 - 2019)", fontsize = 15, fontweight = 'bold', color = berkeley_palette['berkeley_blue'])

    ax1 = fig.add_subplot(133)
    auth_data = df.groupby(pd.Grouper(key = 'created', freq = 'M')).num_authors.mean().to_frame().ffill()
    ax1.scatter(auth_data.index, auth_data, color = berkeley_palette['california_gold'], alpha = 0.5, edgecolor = berkeley_palette['wellman_tile'])
    m, b = np.polyfit(auth_data.index.astype(int), auth_data, 1)
    ax1.plot(auth_data.index, m*auth_data.index.astype(int) + b, color = berkeley_palette['pacific'], linestyle = '--', 
        alpha = 1, linewidth = 3)
    ax1.set_xlabel("Publication Date", fontsize = 12, horizontalalignment = 'right', x = 1.0, color = berkeley_palette['berkeley_blue'])
    ax1.set_ylabel("Mean Author Count", fontsize = 12, horizontalalignment = 'right', y = 1.0, color = berkeley_palette['berkeley_blue'])
    plt.title("Number of Authors per Paper\nArXiV (1993 - 2019)", fontsize = 15, fontweight = 'bold', color = berkeley_palette['berkeley_blue'])

    plt.suptitle(sup_title, fontsize = 20, fontweight = 'bold', color = sup_title_color)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

#---------------------------------------------------------------------------
## Altair Faceted Density Estimate Plot for cs.AI, cs.LG, and stat.ML
#---------------------------------------------------------------------------

def altair_facted_density(sub, domain, range_, xaxis_title, yaxis_title, legend_title, width = 600, height = 150):
    
    #go big or go home!
    alt.data_transformers.disable_max_rows()

    a = alt.Chart(sub[:]).transform_density(
        density = 'count',
        bandwidth = 0.5,
        groupby = ['id'],
        as_ = ['count', 'density'],
        extent = [min(sub['count']), max(sub['count'])]
    ).mark_area(opacity = 0.85).encode(
        alt.X('count:Q', axis = alt.Axis(title = xaxis_title, grid = False)),
        alt.Y('density:Q', axis = alt.Axis(title = 'Density')),
        alt.Row('id:N', header = alt.Header(title = yaxis_title)),
        color = alt.Color('id', 
            scale = alt.Scale(domain = domain, range = range_), 
            legend = alt.Legend(title = legend_title))
    ).properties(width = width, height = height).configure_view(strokeWidth = 0)
    
    return a

#---------------------------------------------------------------------------
## Altair BoxPlot for cs.AI, cs.LG, and stat.ML
#---------------------------------------------------------------------------

def altair_boxplot(sub, xaxis_title, yaxis_title, categories, width = 900, height = 400, outlier_color = berkeley_palette['lawrence'],
    ingroup_color = berkeley_palette['rose_garden'], outgroup_color = berkeley_palette['berkeley_blue'], outlier_shape = "diamond"):

    # go big or go home!
    alt.data_transformers.disable_max_rows()

    range_ = [ingroup_color if (v == 'cs.AI' or v == 'cs.LG' or v == 'stat.ML') else outgroup_color for v in categories]

    a = alt.Chart(sub[:]).mark_boxplot(opacity = 0.90, 
        outliers = alt.MarkConfig(color = outlier_color, shape = outlier_shape)).encode(
        x = alt.X('id:O', axis = alt.Axis(title = xaxis_title, grid = False, labelAngle = 310, labelFontSize = 12)),
        y = alt.Y('count:Q', axis = alt.Axis(title = yaxis_title, grid = True,
            titleFont = "Lato", titleFontWeight = 'bold', titleFontSize = 15,
            titleAnchor = "end", titleAlign = "right")),
        color = alt.Color('id', scale = alt.Scale(domain = categories, range = range_), legend = None
        )
    ).properties(width = width, height = height)

    return a

#---------------------------------------------------------------------------
## Utility function for ngram visualizations
#---------------------------------------------------------------------------

def get_topN_ngrams_over_10_years(df, N):

    def get_yearly_clean_abstracts(df, year):

        target_abstract = df[(pd.DatetimeIndex(df.created).year == year) & (~df.abstract_clean.isna() & (df.language == 'en'))].abstract_clean
        return target_abstract

    top_phrases_over_time = {'year':[], 'ngram':[], 'count':[]}

    # get top N (2,5) ngrams from 2019
    latest_abstracts = get_yearly_clean_abstracts(df, 2019)
    v = CountVectorizer(min_df = 5, max_df = 0.95, analyzer = 'word', ngram_range = (2, 5), token_pattern = '[a-zA-Z\-][a-zA-Z\-]{2,}')
    ngram = v.fit_transform(latest_abstracts)

    vocab = v.vocabulary_
    count_values = ngram.toarray().sum(axis = 0)
    srt = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[:N]

    for y, (c, n) in zip([2019] * N, srt):
        top_phrases_over_time['year'].append(y)
        top_phrases_over_time['ngram'].append(n)
        top_phrases_over_time['count'].append(c)

    latest_ngrams = top_phrases_over_time['ngram'].copy()

    v = CountVectorizer(min_df = 5, max_df = 0.95, analyzer = 'word', ngram_range = (2, 5), token_pattern = '[a-zA-Z\-][a-zA-Z\-]{2,}')

    for i in range(2018, 1998, -1):

        # get the years clean abstracts
        i_abstracts = get_yearly_clean_abstracts(df, i)
            
        # fit the count vectorizer
        ngram = v.fit_transform(i_abstracts)
        count_values = ngram.toarray().sum(axis = 0)
        vocab = v.vocabulary_

        for n in latest_ngrams:
            
            if n in vocab.keys():
                i_cnt = count_values[vocab[n]]
            else:
                i_cnt = 0
            
            top_phrases_over_time['year'].append(i)
            top_phrases_over_time['ngram'].append(n)
            top_phrases_over_time['count'].append(i_cnt)

    df_top_ngrams = pd.DataFrame(top_phrases_over_time)
    tots = df.groupby(by = pd.DatetimeIndex(df.created).year).title.count().to_frame().reset_index().rename(columns = {'title':'total', 'created':'year'})
    df_top_ngrams = df_top_ngrams.merge(tots, how = 'left', left_on = 'year', right_on = 'year')
    df_top_ngrams['reference_ratio'] = df_top_ngrams['count'] / df_top_ngrams['total']

    return df_top_ngrams

#---------------------------------------------------------------------------
## Utility function for visualizing 3 trending ngrams over time
#---------------------------------------------------------------------------

def plot_ngrams_2010_to_present(df_top_ngrams, highlight_ngram_1 = 'neural_network',
    highlight_ngram_2 = 'reinforcement learning', highlight_ngram_3 = 'transfer learning',
    highlight_ngram_1_color = berkeley_palette['berkeley_blue'], highlight_ngram_2_color = berkeley_palette['rose_garden'],
    highlight_ngram_3_color = berkeley_palette['golden_gate'], height = 6, width = 10,
    lowlight_color = berkeley_palette['bay_fog'], xaxis_label = "Calendar Year", yaxis_label = "Ratio of Paper References",
    title = "Trends of N-gram References in AI/ML Papers\n(arXiv : 2010 - 2019)"):

    fig = plt.figure(figsize = (width, height), dpi = 120)
    ax = fig.add_subplot(111)

    plot_ngrams = df_top_ngrams[(df_top_ngrams.year > 2009)].copy()

    years = plot_ngrams.year.sort_values(ascending = True).unique()
    highlight_list = [highlight_ngram_1, highlight_ngram_2, highlight_ngram_3]
    highlight_color = {
        highlight_ngram_1: highlight_ngram_1_color,
        highlight_ngram_2: highlight_ngram_2_color,
        highlight_ngram_3: highlight_ngram_3_color
    }

    for ln in plot_ngrams[(plot_ngrams.year == 2019)].ngram.values:
        sliced_ngram = plot_ngrams[(plot_ngrams.ngram == ln)][['year', 'count', 'reference_ratio']]
        if ln in highlight_list:
            linestyle, alpha, linewidth, linecolor, label = 'dashed', 1, 2, highlight_color[ln], ln
        else:
            linestyle, alpha, linewidth, linecolor, label = 'solid', 0.3, 1, lowlight_color, None

        ax.plot(sliced_ngram.year.values, sliced_ngram['reference_ratio'].values, color = linecolor, linestyle = linestyle, 
            alpha = alpha, linewidth = linewidth, label = label)

    ax.set_xlabel(xaxis_label, fontsize = 12, horizontalalignment = 'right', x = 1.0, color = berkeley_palette['berkeley_blue'])
    ax.set_ylabel(yaxis_label, fontsize = 12, horizontalalignment = 'right', y = 1.0, color = berkeley_palette['berkeley_blue'])
    ax.set_xlim(min(years), max(years))
    ax.set_xticks(years)
    plt.title(title, fontsize = 15, fontweight = 'bold', color = berkeley_palette['berkeley_blue'])

    ax.spines["top"].set_alpha(.0)
    ax.spines["bottom"].set_alpha(.3)
    ax.spines["right"].set_alpha(.0)
    ax.spines["left"].set_alpha(.3)

    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------------
## Utility function for visualizing Poltly 4 n-gram trend (2010-2019)
#---------------------------------------------------------------------------

def plotly_4ngram_trend(df_top_ngrams, labels = ['neural network', 'reinforcement learning', 'machine learning', 'convolutional neural'],
    title = 'Top N-gram Paper Trends', width = 1000, height = 600):

    sub = df_top_ngrams[(df_top_ngrams.year > 2009) & (df_top_ngrams.ngram.isin(labels))].copy()
    x_data = np.vstack((sub.year.sort_values(ascending = True).unique().astype(np.uint16),) * df_top_ngrams.shape[0])

    y_data = []
    for l in labels:
        y_sub = []
        for y in x_data[0]:
            y_sub.append(int(sub[(sub.ngram == l) & (sub.year == y)].reference_ratio * 100))
        y_data.append(y_sub)

    y_data = np.array(y_data)

    colors = [berkeley_palette['berkeley_blue'], berkeley_palette['rose_garden'], 
        berkeley_palette['lawrence'], berkeley_palette['golden_gate']]

    mode_size = [8, 8, 12, 8]
    line_size = [2, 2, 4, 2]


    fig = go.Figure()

    for i in range(0, 4):
        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
        ))

        # endpoints
        fig.add_trace(go.Scatter(
            x=[x_data[i][0], x_data[i][-1]],
            y=[y_data[i][0], y_data[i][-1]],
            mode='markers',
            marker=dict(color=colors[i], size=mode_size[i])
        ))

    fig.update_layout(
        xaxis=dict(
            title = go.layout.xaxis.Title(
                text = "Calendar Year",
                font = dict(
                    family = "Lato",
                    size = 15,
                    color = berkeley_palette['berkeley_blue']
                )
            ),
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Lato',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title = go.layout.yaxis.Title(
                text = "% of Paper References",
                font = dict(
                    family = "Lato",
                    size = 15,
                    color = berkeley_palette['berkeley_blue']
                )
            ),
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        autosize=False,
        width = width,
        height = height,
        margin=dict(
            autoexpand=False,
            l=20,
            r=160,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white'
    )

    annotations = []

    # Adding labels
    for y_trace, label, color in zip(y_data, labels, colors):
        # labeling the left_side of the plot
        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                    xanchor='right', yanchor='middle',
                                    text='{}%'.format(y_trace[len(y_trace)-1]),
                                    font=dict(family='Arial',
                                                size=16, color = color),
                                    showarrow=False))
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.95, y=y_trace[9],
                                    xanchor='left', yanchor='middle',
                                    text='{}%'.format(y_trace[len(y_trace)-1]) + ' ' + label,
                                    font=dict(family='Arial',
                                                size=16, color = color),
                                    showarrow=False))
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.05, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text=title,
                                font=dict(family='Lato',
                                            size=30,
                                            color=berkeley_palette['berkeley_blue']),
                                showarrow=False))
    # Source
    annotations.append(dict(xref='paper', yref='paper', x=1, y=-0.1,
                                xanchor='right', yanchor='top',
                                text='Source:arXiv : cs.AI, cs.ML, stat.ML - ' +
                                    '(2010 - 2019)',
                                font=dict(family='Arial',
                                            size=12,
                                            color='rgb(150,150,150)'),
                                showarrow=False))

    fig.update_layout(annotations=annotations)

    fig.show()


############################################################################
# BOKEH Section
############################################################################

# handle the currently selected article
def selected_code():
    code = """
            var titles = [];
            var authors = [];
            var categories = [];
            var clusters = [];
            var arxivids = [];
            cb_data.source.selected.indices.forEach(index => titles.push(source.data['titles'][index]));
            cb_data.source.selected.indices.forEach(index => authors.push(source.data['authors'][index]));
            cb_data.source.selected.indices.forEach(index => arxivids.push(source.data['arxivids'][index]));
            cb_data.source.selected.indices.forEach(index => categories.push(source.data['categories'][index]));
            cb_data.source.selected.indices.forEach(index => clusters.push(source.data['labels'][index]));
            var title = "<h4>" + titles[0].toString().replace(/<br>/g, ' ') + "</h4>";
            var author = "<p1><b>Authors: </b> " + authors[0].toString().replace(/<br>/g, ' ') + "<br>";
            var arxivid = "<p1><b>arXiv id: </b> " + arxivids[0].toString() + "<br>";
            var category = "<b>P. Category: </b>" + categories[0].toString() + "<br>";
            var cluster = "<b>Cluster: </b>" + clusters[0].toString() + "<br></p1>";
            current_selection.text = title + author + arxivid + category + cluster;
            current_selection.change.emit();
    """
    return code

# handle the keywords and search
def input_callback(plot, source, out_text, topics): 

    # slider call back for cluster selection
    callback = CustomJS(args=dict(p=plot, source=source, out_text=out_text, topics=topics), code="""
				var key = text.value;
				key = key.toLowerCase();
				var cluster = slider.value;
                var data = source.data; 
                var i = 0;

                var x = data['x'];
                var y = data['y'];
                var x_backup = data['x_backup'];
                var y_backup = data['y_backup'];
                var labels = data['desc'];
                var abstract = data['abstract'];
                var titles = data['titles'];
                var authors = data['authors'];
                var categories = data['categories'];
                if (cluster == '15') {
                    out_text.text = 'Keywords: Slide to specific cluster to see the keywords.';
                    for (i = 0; i < x.length; i++) {
						if(abstract[i].includes(key) || 
						titles[i].includes(key) || 
						authors[i].includes(key) || 
						categories[i].includes(key)) {
							x[i] = x_backup[i];
							y[i] = y_backup[i];
						} else {
							x[i] = undefined;
							y[i] = undefined;
						}
                    }
                }
                else {
                    out_text.text = 'Keywords: ' + topics[Number(cluster)];
                    for (i = 0; i < x.length; i++) {
                        if(labels[i] == cluster) {
							if(abstract[i].includes(key) || 
							titles[i].includes(key) || 
							authors[i].includes(key) || 
							categories[i].includes(key)) {
								x[i] = x_backup[i];
								y[i] = y_backup[i];
							} else {
								x[i] = undefined;
								y[i] = undefined;
							}
                        } else {
                            x[i] = undefined;
                            y[i] = undefined;
                        }
                    }
                }
            source.change.emit();
            """)
    return callback

def bokeh_load_clusters_plot(df, topics, output_filename = './reports/bokeh/t-sne_arxiv_abstracts.html'):

    output_notebook()
    # target labels
    y_labels = df.cluster.values

    # data sources
    source = ColumnDataSource(data=dict(
        x= df.tsne_X.values, 
        y= df.tsne_Y.values,
        x_backup = df.tsne_X.values,
        y_backup = df.tsne_Y.values,
        desc= y_labels, 
        titles= df.title.values,
        authors = df.author_text.values,
        categories = df.primary_cat.values,
        abstract = df.abstract.values,
        labels = ["C-" + str(x) for x in y_labels],
        arxivids = df.arxiv_id.values
        ))

    # hover over information
    hover = HoverTool(tooltips=[
        ("Title", "@titles{safe}"),
        ("arXiv id", "@arxivids{safe}"),
        ("Author(s)", "@authors{safe}"),
        ("Category", "@categories{safe}"),
        ("Cluster", "@labels{safe}"),
        ("Abstract", "@abstract{safe}")
    ],
    point_policy="follow_mouse")

    # map colors
    mapper = linear_cmap(field_name='desc', 
                        palette=Category20[20],
                        low=min(y_labels) ,high=max(y_labels))

    # prepare the figure
    plot = figure(plot_width=800, plot_height=500, 
            tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap'], 
            title="Clustering of arXiv [cs.AI, cs.LG, stat.ML] Literature from 1993-2019 with PCA, t-SNE and K-Means (K=15)", 
            toolbar_location="above")

    # plot settings
    plot.scatter('x', 'y', size=5, 
            source=source,
            fill_color=mapper,
            line_alpha=0.3,
            line_color="black",
            #legend = 'labels')
            legend_group = 'labels')

    plot.legend.background_fill_alpha = 0.6

    # Keywords
    text_banner = Paragraph(text= 'Keywords: Slide to specific cluster to see the keywords.', height=45)
    input_callback_1 = input_callback(plot, source, text_banner, topics)

    # currently selected article
    div_curr = Div(text="""Click on a plot to see the article summary.""",height=150)
    callback_selected = CustomJS(args=dict(source=source, current_selection=div_curr), code=selected_code())
    taptool = plot.select(type=TapTool)
    taptool.callback = callback_selected

    # WIDGETS
    #slider = Slider(start=0, end=15, value=15, step=1, title="Cluster #", js_event_callbacks = {'on_change': [input_callback_1]})
    slider = Slider(start=0, end=15, value=15, step=1, title="Cluster #")
    #keyword = TextInput(title="Search:", js_event_callbacks = {'on_change': [input_callback_1]})
    keyword = TextInput(title="Abstract Search:")

    # pass call back arguments
    input_callback_1.args["text"] = keyword
    input_callback_1.args["slider"] = slider

    slider.js_on_change('value', input_callback_1)
    keyword.js_on_change('value', input_callback_1)

    # STYLE
    slider.sizing_mode = "stretch_width"
    slider.margin=15

    keyword.sizing_mode = "scale_both"
    keyword.margin=15

    div_curr.style={'color': '#BF0A30', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
    div_curr.sizing_mode = "scale_both"
    div_curr.margin = 20

    text_banner.style={'color': '#0269A4', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
    text_banner.sizing_mode = "scale_both"
    text_banner.margin = 20

    plot.sizing_mode = "scale_both"
    plot.margin = 5

    r = row(div_curr,text_banner)
    r.sizing_mode = "stretch_width"

    # LAYOUT OF THE PAGE
    l = layout([
        [slider, keyword],
        [text_banner],
        [div_curr],
        [plot],
    ])
    l.sizing_mode = "scale_both"

    # show
    output_file(filename = output_filename, title = 'arXiv Abstract Clusters (K=15)')
    save (l)
    del l

    return
