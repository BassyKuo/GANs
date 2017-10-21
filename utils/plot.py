# Code derived from https://github.com/musyoku/unrolled-gan
import sys
import numpy as np
import pylab, os
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("white")

def plot_kde(data, dir='plot', filename="kde", color="Greens", suffix=''):
    if os.path.exists(dir) is False:
        os.mkdir(dir)
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    bg_color  = sns.color_palette(color, n_colors=256)[0]
    ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
    ax.set_axis_bgcolor(bg_color)
    # ax.set_title('{}'.format(filename))
    kde = ax.get_figure()
    pylab.xlim(-4, 4)
    pylab.ylim(-4, 4)
    pylab.title('{}\n{}'.format(filename, suffix), fontsize=24)
    kde.savefig(os.path.join(dir, "{}{}.png".format(filename, suffix)))

def plot_scatter(data, dir='plot', filename="scatter", color="blue", suffix=''):
    if os.path.exists(dir) is False:
        os.mkdir(dir)
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
    pylab.xlim(-4, 4)
    pylab.ylim(-4, 4)
    pylab.title('{}\n{}'.format(filename, suffix), fontsize=24)
    pylab.savefig(os.path.join(dir, "{}{}.png".format(filename, suffix)))

def plot_heatmap(data, value, dir='plot', filename="heatmap", prob=True):
    assert len(data.shape) == 2
    assert data.shape[1] == 2
    assert data.shape[0] == value.shape[0]
    assert len(value.shape) in [1,2]
    if os.path.exists(dir) is False:
        os.mkdir(dir)
    value = value.reshape(-1)
    xy = np.rint(data*10).astype(int)
    idx = (xy >= -40) * (xy < 40)
    M = np.zeros([80,80])
    for r, ((tx,ty), (i,j)) in enumerate(zip(idx, xy)):
        # print (r, tx, ty, i, j, value[r])
        if tx*ty == 1:
            M[79-(j+40),i+40] = value[r]
    # print ("M: ", M.shape)
    labels = list(np.arange(-4,4).astype(int))
    labellist = [''] * 80
    for i,l in enumerate(labels):
        labellist[i*10] = l
    labellist[-1] = 4

    fig = pylab.gcf()
    fig.set_size_inches(20.0, 16.0)
    pylab.clf()
    if prob == True:
        ax = sns.heatmap(M, cbar=True, xticklabels=labellist, yticklabels=labellist[::-1], cbar_kws={'ticks':[0,.2,.4,.6,.8,1]}, vmax=1, vmin=0)
    else:
        ax = sns.heatmap(M, cbar=True, xticklabels=labellist, yticklabels=labellist[::-1])
    hmap = ax.get_figure()
    pylab.title('{}\n{}'.format(filename, suffix), fontsize=24)
    hmap.savefig(os.path.join(dir, "{}{}.png".format(filename, suffix)))

def plot_heatmap_fast(value, dir='plot', filename="heatmap", prob=True, suffix=''):
    '''
    ==For example==
    value[0,0] = (x0,y0)
    value[0,1] = (x0,y1)
    value[0,2] = (x0,y2)
    ...
    value[4,4] = (x4,y4)
    '''
    check = np.zeros([])
    assert len(value.shape) == 2
    assert type(value) == type(check)
    if os.path.exists(dir) is False:
        os.mkdir(dir)
    labels = list(np.arange(-4,4).astype(int))
    labellist = [''] * 80
    for i,l in enumerate(labels):
        labellist[i*10] = l
    labellist[-1] = 4

    fig = pylab.gcf()
    fig.set_size_inches(20.0, 16.0)
    pylab.clf()
    if prob == True:
        if value.min() < 0 and value.max() > 1:
            value = 1. / (1+np.exp(-value))
        ax = sns.heatmap(value, cbar=True, xticklabels=labellist, yticklabels=labellist[::-1], cbar_kws={'ticks':[0,.2,.4,.6,.8,1]}, vmax=1, vmin=0)
    else:
        ax = sns.heatmap(value, cbar=True, xticklabels=labellist, yticklabels=labellist[::-1])
    hmap = ax.get_figure()
    pylab.title('{}\n{}'.format(filename, suffix), fontsize=24)
    hmap.savefig(os.path.join(dir, "{}{}.png".format(filename, suffix)))


def main(plot_dir, num_mixture):
    num_samples = 10000
    print ('Plot gaussian_mixture_circle ...1')
    samples_true = sampler.gaussian_mixture_circle(num_samples, num_mixture, scale=1, std=0.2)
    plot_scatter(samples_true, plot_dir, "scatter_true_1")
    plot_kde(samples_true, plot_dir, "kde_true_1")

    print ('Plot gaussian_mixture_circle ...2')
    samples_true = sampler.gaussian_mixture_circle(num_samples, num_mixture, scale=2, std=0.2)
    plot_scatter(samples_true, plot_dir, "scatter_true_2")
    plot_kde(samples_true, plot_dir, "kde_true_2")

    print ('Plot gaussian_mixture_double_circle ...')
    samples_true_2 = sampler.gaussian_mixture_double_circle(num_samples, num_mixture, scale=2, std=0.2)
    plot_scatter(samples_true_2, plot_dir, "scatter_true_double")
    plot_kde(samples_true_2, plot_dir, "kde_true_double")

if __name__ == "__main__":
    import sampler
    try:
        plot_dir = sys.argv[1]
    except IndexError:
        plot_dir = 'plot'
    try:
        num_mixture = sys.argv[2]
    except IndexError:
        num_mixture = 8
    main(str(plot_dir), int(num_mixture))
