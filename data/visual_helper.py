#!/usr/bin/env python
# coding: utf-8

from visdom import Visdom
import numpy as np

viz = Visdom(port=8093)


class Visual(object):
    """visdom wraper"""
    def __init__(self, win=None, lines=None, new_win=True):
        self.index = 0
        self.win = win
        self.lines = lines
        if new_win:
            viz.line(X=np.array([self.index]),
                    Y=np.array([0]),
                    win=win,
                    name=lines[0],
                    opts={'title': win},
                    )
        else:
            viz.line(X=np.array([self.index]),
                    Y=np.array([0]),
                    win=win,
                    name=lines[0],
                    )
        for line_name in lines[1:]:
            viz.line(X=np.array([0]),
                    Y=np.array([0]),
                    win=win,
                    name=line_name,
                    update='append'
                    )
        self.index += 1

    def plot(self, line_val):
        for k,v in line_val.items():
            viz.line(X=np.array([self.index]),
                    Y=np.array([v]),
                    win=self.win,
                    name=k,
                    update='append',
                    )
        self.index += 1
