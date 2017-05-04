#!/usr/bin/env

import matplotlib.pyplot as plt
import numpy as np
import glob
from string import punctuation
from collections import Counter
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import itertools
from scipy import stats
from word_counter import mfw_counter
from preprocess import vectorize

def rolling_delta(sample_length, feat_amount, invalid_words, step_size):

    tls.set_credentials_file(username='jedgusse.degussem', api_key='qn9k3mb0ic')
    plotname = "Nicholas_VS_Bernard"

    # First we produce a standard centroid vector, which serves as a benchmark against which we can test every sample separately
    # We can expect that the figure will reveal great peaks or lows whenever there has been an "intrusion" by a collaborator
    # We express the frequencies of words in relative frequencies, so we first turn the raw counts to relative frequencies.
    # We also weight the relative frequencies by standard deviation
    # See "Collaborative Authorship: Conrad, Ford and Rolling Delta"

    # First collect benchmark corpus, and vectorize the corpus into raw counts
    authors, titles, texts, raw_counts, features = vectorize("/Users/jedgusse/stylofactory/corpora/rolling_delta/benchmark_corpus", sample_length, feat_amount, invalid_words)

    # We first turn our raw counts into relative frequencies
    relative_vectors = [vector / np.sum(vector) for vector in raw_counts]

    # We produce a standard deviation vector, that will later serve to give more weight to highly changeable words and serves to
    # boost words that have a low frequency. This is a normal Delta procedure.
    # We only calculate the standard deviation on the benchmark corpus, since that is the distribution against which we want to compare
    stdev_vector = np.std(relative_vectors, axis = 0)

    # We make a centroid vector for the benchmark corpus
    centroid_vector = np.mean(relative_vectors, axis=0)

    # Now we have turned all the raw counts of the benchmark corpus into relative frequencies, and there is a centroid vector
    # which counts as a standard against which the test corpus can be compared.

    data = []

    for text_index, filename in enumerate(glob.glob("/Users/jedgusse/stylofactory/corpora/rolling_delta/test_corpus/*")):
    author = filename.split("/")[-1].split(".")[0].split("_")[0]
    title = filename.split("/")[-1].split(".")[0].split("_")[1]

    print("::: rolling delta on text {} :::".format(title))

    bulk = []

    fob = open(filename)
    text = fob.read()
    for word in text.rstrip().split():

        for char in word:
        if char in punctuation:
            word = word.replace(char, "")
        word = word.lower()
        bulk.append(word)

    # We now divide the test corpus in the given sample lengths, taking into account the step_size of overlap
    # This is the "shingling" procedure, where we get overlap, where we get windows

    steps = np.arange(0, len(bulk), step_size)
    step_ranges = []

    step_sized_samples = []
    for each_begin in steps:
        sample_range = range(each_begin, each_begin+sample_length)
        step_ranges.append(sample_range)
        text_sample = []
        for index, word in enumerate(bulk):
        if index in sample_range:
            text_sample.append(word)
        step_sized_samples.append(text_sample)

    # Now we change the samples to numerical values, using the features as determined in code above
    # Only allow text samples that have desired sample length

    window_vectors = []
    for text_sample in step_sized_samples:
        if len(text_sample) == sample_length:
        vector = []
        counter = Counter(text_sample)
        for feature in features:
            vector.append(counter[feature])
        window_vectors.append(vector)
    window_vectors = np.asarray(window_vectors)

    # We turn the raw counts into relative frequencies

    relativized = [vector / np.sum(vector) for vector in window_vectors]

    delta_scores = []
    for vector in relativized:
        delta_distances = np.mean(np.absolute(centroid_vector - vector) / stdev_vector)
        delta_score = np.mean(delta_distances)
        delta_scores.append(delta_score)

    trace = go.Scatter(
        x = [graphthing[-1] for graphthing in step_ranges],
        y = [each for each in delta_scores],
        name='{}'.format(filename.split("/")[-1]),
        line = dict(width = 1))
    data.append(trace)

    layout = go.Layout(title='Double Y Axis Example', yaxis=dict(title='yaxis title'), yaxis2=dict(title='yaxis2 title', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying='y', side='right'))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(data, filename='{}'.format(plotname))
