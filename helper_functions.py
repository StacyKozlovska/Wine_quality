"""
Helper functions necessary for the data analysis of Podcast Reviews.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
from varname import nameof
from pprint import pprint
import textwrap
from scipy import stats


sns.set_style("darkgrid")

xticklabels_d = {
    "horizontalalignment": "right",
    "fontweight": "light",
    "fontsize": "x-large",
}

def two_sample_summary_stats(group_1, group_2, statistic: str, col_1_name: str, col_2_name: str):
    """
    Function to get summary statistic for two dataframes and 
    the difference in that statistic between two groups.
    """
    group0_statistic = getattr(group_1, statistic)()
    group1_statistic = getattr(group_2, statistic)()

    diff_statistic = group0_statistic - group1_statistic

    result_df = pd.DataFrame({
        f"{col_1_name}_{statistic}": group0_statistic,
        f"{col_2_name}_{statistic}": group1_statistic,
        f"{col_1_name}_sub_{col_2_name}_{statistic}_diff": diff_statistic
    })

    return result_df


def plot_sns_barplot(
    data,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    title: str,
    hue=None,
    xtick_rot: int = 65,
    max_len_xtick_labels: int = 25,
    xticklabels: dict = xticklabels_d,
    my_figsize: (int, int) = (10, 7),
):
    """
    Function to automate seaborn
    barplot plotting.
    """
    # Figure Size
    fig = plt.figure(figsize=my_figsize)

    # Bar Plot
    ax = sns.barplot(x=data[x], y=data[y], hue=hue)
    f = lambda x: textwrap.fill(x.get_text(), max_len_xtick_labels)
    ax.set_xticklabels(map(f, ax.get_xticklabels()), rotation=xtick_rot, **xticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.plot()


def plot_sns_countplot(
    data,
    x: str,
    plot_order,
    x_label: str,
    y_label: str,
    title: str,
    hue=None,
    xtick_rot: int = 65,
    max_len_xtick_labels: int = 25,
    xticklabels: dict = xticklabels_d,
    my_figsize: (int, int) = (10, 7),
):
    """
    Function to automate seaborn
    countplot plotting.
    """
    plt.figure(figsize=my_figsize)
    ax = sns.countplot(data=data, x=x, order=plot_order, hue=hue)
    f = lambda x: textwrap.fill(x.get_text(), max_len_xtick_labels)
    ax.set_xticklabels(map(f, ax.get_xticklabels()), rotation=xtick_rot, **xticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.plot()
    return ax


def plot_sns_jointplot(
    data, x: str, y: str, title: str, xlim=(-20, 850), ylim=(3, 5.1), my_figsize=(8, 5)
):
    """
    Function to automate seaborn
    jointplot plotting.
    """
    g = sns.JointGrid(data, x=x, y=y)
    g.plot_joint(sns.scatterplot, s=100, alpha=0.5)
    g.ax_marg_x.set_xlim(*xlim)
    g.ax_marg_y.set_ylim(*ylim)
    g.plot_marginals(sns.histplot, kde=True)
    g.fig.set_size_inches(my_figsize)
    g.fig.suptitle(title)

    g.fig.show()


def visualize_violinplot(df, x: str, y: str, hue: str = None):
    """
    Function to plot a violin plot with seaborn.
    """
    # Create the violin plot
    sns.violinplot(x=x, y=y, data=df, hue=hue)

    # Set the plot title and axes labels
    plt.title(f"{y.capitalize()} distribution by {x.capitalize()}")
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())

    if hue is not None:
        plt.legend(title=hue, loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.title(
            f"{y.capitalize()} distribution by {x.capitalize()} and {hue.capitalize()}"
        )
    # Show the plot
    plt.show()


def plot_count_percent_barplots_by_category(
    my_df, cat_col: str, my_col: str, my_title: str, my_order=None
):
    """
    Function to visualize two plots side by side.
    The first plot shows the total count for each category.
    The second plot shows the shares for each category.
    """

    fig = plt.figure(figsize=(16, 8))
    grid = GridSpec(1, 2)

    ax1 = fig.add_subplot(grid[0, 0])

    # Set the color palette for the countplot
    palette = {0: "darkgrey", 1: "steelblue"}

    sns.countplot(
        data=my_df.dropna(),
        x=cat_col,
        order=my_order,
        hue=my_col,
        palette=palette,
        ax=ax1,
        width=0.8,
    )
    ax1.set(xlabel=cat_col.capitalize(), ylabel="Count")

    ax2 = fig.add_subplot(grid[0, 1])

    # Calculate the share of positive ratings for each category
    share_positive = my_df.groupby(cat_col)[my_col].mean()

    if my_order is not None:
        share_positive = share_positive.loc[my_order].sort_values(ascending=False)

        # Calculate the share of negative ratings for each category
    share_negative = 1 - share_positive

    # Plot the stacked bars
    ax2.bar(
        share_positive.index,
        share_positive,
        color="steelblue",
        label="Positive",
    )
    ax2.bar(
        share_negative.index,
        share_negative,
        bottom=share_positive,
        color="darkgrey",
        label="Negative",
    )

    ylabel_name = my_col.replace("_", " ")

    ax2.set(xlabel=cat_col.capitalize(), ylabel=f"Share of {ylabel_name.capitalize()}")
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Rotate the x-axis labels by 45 degrees
    ax1.tick_params(axis="x", labelrotation=45)
    ax2.tick_params(axis="x", labelrotation=45)

    fig.suptitle(my_title, fontsize=16)

    plt.subplots_adjust(wspace=0.3)

    plt.show()


def visualize_two_violin_boxplots_stats(df, features_df, x_label: str, sample_1_val, sample_2_val):
    """
    Function to visualize half violin half boxplots for 
    two sample distributions with a text box in the middle
    containing info about the difference in means, medians, modes
    and variances between the two samples.
    """
    for i, c in enumerate(features_df.columns):
        plt.figure(i, figsize=(7, 4))
        
        # Create violin plot (left half)
        sns.violinplot(
            x=x_label,
            y=f"{c}",
            data=df,
            hue=x_label,
            split=True,
            inner="quart",
            linewidth=1,
        )
        
        # Create box plot (right half)
        sns.boxplot(
            x=x_label,
            y=f"{c}",
            data=df,
            width=0.2,
            showcaps=False,
            boxprops={'facecolor':'None', 'edgecolor':'black'},
            showfliers=False,
            whiskerprops={'linewidth': 2},
            saturation=1,
            zorder=10
        )
        # Calculate the difference in means, medians, and modes between the two groups
        group0_data = df[df[x_label] == sample_1_val][c]
        group1_data = df[df[x_label] == sample_2_val][c]
        mean_diff = group0_data.mean() - group1_data.mean()
        median_diff = group0_data.median() - group1_data.median()
        mode_diff = group0_data.mode().values[0] - group1_data.mode().values[0]
        
        # Calculate the difference in variances between the two groups
        var_diff = group0_data.var() - group1_data.var()

        # Annotate the plot with the calculated differences
        text_box = f"Group {sample_1_val} - Group {sample_2_val} difference:\nMean Diff: {mean_diff:.2f}\nMedian Diff: {median_diff:.2f}\nMode Diff: {mode_diff:.2f}\nVar Diff: {var_diff:.2f}"
        plt.text(0.5, 0.5, text_box, transform=plt.gca().transAxes, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=1, boxstyle='round,pad=0.5'))
        
        plt.title(f"Half Violin and Half Box Plots for {c} based on {x_label}")
        plt.xlabel(x_label)
        plt.ylabel(f"{c}")
    
    plt.show()


def q_based_floor_cap(my_df, features_df, lower_upper_levels):
    """
    Function to perform quantile based flooring and capping.
    """
    pd.options.mode.chained_assignment = None
    floor_cap_data = my_df.copy(deep=True)
    for col in features_df.columns:
        percentiles = floor_cap_data.loc[:, col].quantile(lower_upper_levels).values
        floor_cap_data.loc[:, col][floor_cap_data[col] <= percentiles[0]] = percentiles[0]
        floor_cap_data.loc[:, col][floor_cap_data[col] >= percentiles[1]] = percentiles[1]

    return floor_cap_data


def test_two_sample_ttest_assumptions(group_1, group_2, feature):
    """
    Function to check whether the assumptions for a two-sample
    t-test are met.
    """
    group0_data = group_1[feature]
    group1_data = group_2[feature]


    # Shapiro-Wilk test for normality
    shapiro_stat_group0, shapiro_pvalue_group0 = stats.shapiro(group0_data)
    shapiro_stat_group1, shapiro_pvalue_group1 = stats.shapiro(group1_data)
    
    # Levene's test for equal variances
    levene_stat, levene_pvalue = stats.levene(group0_data, group1_data)

    print(f"Results of the tests for assumptions for 2-sample t-tests for groups in {feature}:")
    print("-"*70)
    
    print(
        "Shapiro-Wilk Test - Group 1: Statistic =",
        shapiro_stat_group0,
        "P-value =",
        shapiro_pvalue_group0,
    )
    print(
        "Shapiro-Wilk Test - Group 2: Statistic =",
        shapiro_stat_group1,
        "P-value =",
        shapiro_pvalue_group1,
    )
    print("Levene's Test - P-value =", levene_pvalue)
    
    if (
        shapiro_pvalue_group0 > 0.05
        and shapiro_pvalue_group1 > 0.05
        and levene_pvalue > 0.05
    ):
        print(
            "Both groups pass the normality and equal variance assumptions for the t-test."
        )
    else:
        print("One or both groups may not meet the assumptions for the t-test.")
    
    print("-" * 50)
    
    # Check normality using scipy.stats.normaltest
    normaltest_stat_group0, normaltest_pvalue_group0 = stats.normaltest(group0_data)
    normaltest_stat_group1, normaltest_pvalue_group1 = stats.normaltest(group1_data)
    
    print(
        "D'Agostino and Pearson's omnibus Test for Normality - Group 1: Statistic =",
        normaltest_stat_group0,
        "P-value =",
        normaltest_pvalue_group0,
    )
    print(
        "D'Agostino and Pearson's omnibus Test for Normality - Group 2: Statistic =",
        normaltest_stat_group1,
        "P-value =",
        normaltest_pvalue_group1,
    )
    
    alpha = 0.05
    if normaltest_pvalue_group0 > alpha and normaltest_pvalue_group1 > alpha:
        print("Both groups are approximately normally distributed.")
    else:
        print("One or both groups may not be normally distributed.")
    
    print("-" * 50)
    
    # Create Q-Q plots for both groups
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    stats.probplot(group0_data, plot=plt)
    plt.title(f"Q-Q Plot for {feature} - Group 1")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    
    plt.subplot(2, 1, 2)
    stats.probplot(group1_data, plot=plt)
    plt.title(f"Q-Q Plot {feature} - Group 2")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    
    plt.tight_layout()
    plt.show()


def mannwhitneyu_two_sided_test(group_1, group_2, feature):
    """
    Function to check whether the medians of two distributions are
    significantly different with the help of the Mann-Whitney U test.
    """
    group0_data = group_1[feature]
    group1_data = group_2[feature]
    
    statistic, p_value = stats.mannwhitneyu(
    group0_data, group1_data, alternative="two-sided"
    )
    
    print("Mann-Whitney U Statistic:", statistic)
    print("P-value:", round(p_value, 4))
    
    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference between the two groups.")
    else:
        print("There is no significant difference between the two groups.")


def two_sided_ttest(group_1, group_2, feature):
    """
    Function to check whether the medians of two distributions are
    significantly different with the help of the Mann-Whitney U test.
    """
    group0_data = group_1[feature]
    group1_data = group_2[feature]
    
    # Perform the two-sample t-test
    t_statistic, p_value = stats.ttest_ind(group0_data, group1_data)
    
    print("Two-Sample T-Test Results:")
    print("T-Statistic:", t_statistic)
    print("P-value:", round(p_value, 4))
    
    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference in means between the two groups.")
    else:
        print("There is no significant difference in means between the two groups.")