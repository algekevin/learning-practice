import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
from datetime import datetime

class PandasDataCh8:
    def __init__(self):
        #self.plots_pt1()
        self.ticks_labels()
        #self.plots()

        plt.show()

        # This chapter is done. Chapter 9 continues on to focus on groupby() and other tecniques.

    def plots_pt1(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)  # 2x2 figure, this adds the subplot in slot 1
        ax2 = fig.add_subplot(2, 2, 2)  # ... slot 2
        ax3 = fig.add_subplot(2, 2, 3)  # ... slot 3
        ax4 = fig.add_subplot(2, 2, 4)  # ... slot 4

        plt.plot(np.random.randn(50).cumsum(), "k--")  # Picks the last slot of the subplots, ie ax4
        _ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
        ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))

        # fig, axes = plt.subplots(2, 3)
        # print(fig, '\n', axes)

        # ax.plot(x, y, 'g--') is the same as ax.plot(x, y, linestyle='--', color='g')
        # Get any colors you want by doing something liek color='#CECECE'

        # Line can have markers for the actual data points:
        ax3.plot(np.random.randn(30).cumsum(), 'go--')  # Explicitly as ..., linestyle='--', marker='o', color='g'

    def ticks_labels(self):
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)

        ticks = ax.set_xticks([0, 250, 500, 750, 1000])
        labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                                rotation=30, fontsize='small')

        ax.set_title('PyPlot')
        ax.set_xlabel('Stages')

        ax.plot(np.random.randn(1000).cumsum(), 'g--', label='one')
        ax.plot(np.random.randn(1000).cumsum(), color='orange', label='two')
        ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
        ax.legend(loc='best')

        ax2 = fig.add_subplot(2, 2, 2)
        rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='orange', alpha=0.3)
        circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
        poly = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                           color='g', alpha=0.3)
        ax2.add_patch(rect)
        ax2.add_patch(circ)
        ax2.add_patch(poly)

    def plots(self):
        s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
        #s.plot()

        df = DataFrame(np.random.randn(10, 4).cumsum(0),
                       columns=['A', 'B', 'C', 'D'],
                       index=np.arange(0, 100, 10))
        #df.plot(kind='bar') #barh for horizontal bars

        comp1 = np.random.normal(0, 1, size=1000)  # N(0,1)
        comp2 = np.random.normal(10, 2, size=1000) # N(10,4)
        values = Series(np.concatenate([comp1, comp2]))
        values.hist(bins=500, alpha=0.3, normed=True, color='orange')
        values.plot(kind='kde', style='k--')

PandasDataCh8()