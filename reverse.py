#!/usr/bin/env python

from TreeModel import XGBModel

with open('model.bin', 'rb') as f:
    buffer = f.read()
    model = XGBModel.load(buffer)

xs = []
with open('input.txt') as f:
    count = 0
    for line in f:
        data = line.split(' ')
        indices = []
        values = []
        for t in data[1:]:
            i, v = t.split(':')
            indices.append(int(i))
            values.append(float(v))
        x = model.predictLeafInst(indices, values)
        xs.append(x)
        count+=1
