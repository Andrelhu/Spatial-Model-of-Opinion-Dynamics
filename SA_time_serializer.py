import pandas as pd
import matplotlib.pyplot as plt

initial = 'bayonetta2_df'

initial_df = pd.load(initial)
data = {}
target = initial_df.day
target2 = target.join(initial_df.polarity)

for i in target2.values:
    if str(i[0]) in data:
        if float(i[1]) > 0:
            data[str(i[0])][0] += float(abs(i[1]))
        elif float(i[1]) == 0:
            data[str(i[0])][1] += 1
        elif float(i[1]) < 0:
            data[str(i[0])][2] += float(abs(i[1]))
    else:
        if float(i[1]) > 0:
            data.update({str(i[0]):[float(abs(i[1])),0,0]})
        elif float(i[1]) == 0:
            data.update({str(i[0]):[0,1,0]})
        elif float(i[1]) < 0:
            data.update({str(i[0]):[0,0,float(abs(i[1]))]})

data_df = pd.DataFrame.from_dict(data,orient='index')
data_df.columns = ['pos','neu','neg']
fig = plt.figure()
plt.plot()