#!/usr/bin/env

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colours = {'1': '#6190FF', '2': '#A4CDFF', '3': '#FF4F46', '4': '#FF9181', '5': '#FFC3A4', 'n': '#3FD121', 'hugo': '#D157B5', 'h': '#D157B5',
       'GuibNog': '#6190FF', 'Alain': '#000000', 'GilPoit': '#dc1818', 'AnsLaon': '#dc1818', '...': '#dc1818', 'a': '#D157B5', 'JohnSal': '#D157B5', 'WilConch': '#3FD121'}

#div = #B6B5C9

def principal_components_analysis(vectors, authors, titles, features, show_samples, show_loadings):

    pca = PCA(n_components=2)
    X_bar = pca.fit_transform(vectors)
    var_exp = pca.explained_variance_ratio_
    comps = pca.components_
    comps = comps.transpose()
    loadings = pca.components_.transpose()
    vocab_weights_p1 = sorted(zip(features, comps[:,0]), key=lambda tup: tup[1], reverse=True)
    vocab_weights_p2 = sorted(zip(features, comps[:,1]), key=lambda tup: tup[1], reverse=True)

    print("Explained variance: ", sum(pca.explained_variance_ratio_)*100)

    if show_samples == 'yes':

    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(111)
    x1, x2 = X_bar[:,0], X_bar[:,1]

    exclusion_list = []

    ax.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
    for p1, p2, a, title in zip(x1, x2, authors, titles):
        if a not in exclusion_list:
        ax.text(p1, p2, title[:2] + '_' + title.split("_")[1], ha='center',
        va='center', color=colours[a], fontdict={'size': 7})
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    #for p1, p2, a, title in zip(x1, x2, authors, titles):
        #ax.text(p1, p2, title[:3] + '_' + title.split("_")[1], ha='center',
            #va='center', color=colours[a], fontdict={'size': 13})

    # Align the axes

    def align_yaxis(ax1, v1, ax2, v2):
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny+dy, maxy+dy)
    def align_xaxis(ax1, v1, ax2, v2):
        x1, _ = ax1.transData.transform((v1, 0))
        x2, _ = ax2.transData.transform((v2, 0))
        inv = ax2.transData.inverted()
        dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0))
        minx, maxx = ax2.get_xlim()
        ax2.set_xlim(minx+dx, maxx+dx)

    #align_xaxis(ax1, 0, ax2, 0)
    #align_yaxis(ax1, 0, ax2, 0)

    # Legend settings (code for making a legend)

    """brev_patch = mpatches.Patch(color=colours['1'], label='Bernard\'s intra corpus (brevis)')
    new_patch = mpatches.Patch(color=colours['2'], label='Bernard\'s intra corpus (perfectum additions)')
    pre_patch = mpatches.Patch(color=colours['3'], label='Bernard\'s extra corpus (pre-1140)')
    mid_patch = mpatches.Patch(color=colours['4'], label='Bernard\'s extra corpus (1140-1145)')
    post_patch = mpatches.Patch(color=colours['5'], label='Bernard\'s extra corpus (post-1145)')
    nic_patch = mpatches.Patch(color=colours['n'], label='Nicholas\' letters and sermons')
    hugo_patch = mpatches.Patch(color=colours['hugo'], label='Nicholas\' letters and sermons')
    sc_patch = mpatches.Patch(color=colours['sc'], label='Nicholas\' letters and sermons')
    div_patch = mpatches.Patch(color=colours['div'], label='Nicholas\' letters and sermons')
    dub_patch = mpatches.Patch(color=colours['lec'], label='Nicholas\' letters and sermons')

    #plt.legend(handles=[brev_patch, new_patch, pre_patch, mid_patch, post_patch, nic_patch], loc=2, prop={'size':9})
    #plt.legend(handles=[nic_patch, hugo_patch, sc_patch, div_patch, dub_patch], loc=2, prop={'size':9})"""

    # Plotting dotted origin lines

    plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
    plt.axvline(x=0, ls="--", lw=0.5, c='0.75')

    if show_loadings == 'yes':
        ax2 = ax.twinx().twiny()
        l1, l2 = loadings[:,0], loadings[:,1]
        ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
        for x, y, l in zip(l1, l2, features):
        ax2.text(x, y, l, ha='center', va="center", color="black",
        fontdict={'family': 'Arial', 'size': 12})
    elif show_loadings == 'no':
        print("No loadings in PCA")

    plt.show()
    fig.savefig("/Users/jedgusse/stylofactory/output/fig_output/pcafig.pdf", transparent=True, format='pdf')
    # Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

    elif show_samples == 'no':

    fig = plt.figure(figsize=(80,12))

    ax2 = fig.add_subplot(111)
    l1, l2 = loadings[:,0], loadings[:,1]
    ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none')
    for x, y, l in zip(l1, l2, features):
        ax2.text(x, y, l, ha='center', va='center', color='black',
        fontdict={'family': 'Arial', 'size': 10})

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')

    plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
    plt.axvline(x=0, ls="--", lw=0.5, c='0.75')

    plt.show()
    fig.savefig("/Gebruikers/jedgusse/stylofactory/output/fig_output/pcafig.pdf", transparent=True, format='pdf')
    # Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

    # Returns the explained variance ratio
    return var_exp[0] + var_exp[1]