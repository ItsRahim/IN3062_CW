def visualise():
    colours = ['#364F6B', '#3FC1C9', '#F5F5F5', '#FC5185']

figure = plt.figure(figsize=(15, 15), dpi=100)
gs = figure.add_gridspec(3, 3)
gs.update(wspace=0.25, hspace=0.5)

ax0 = figure.add_subplot(gs[0, 0])
ax1 = figure.add_subplot(gs[0, 1])

figure.patch.set_facecolor('#F5F5F5')
ax0.set_facecolor('#F5F5F5')
ax1.set_facecolor('#F5F5F5')

healthy_person = df[df['stroke'] == 0]
unhealthy_person = df[df['stroke'] == 1]

gender = ['Male', 'Female']
smoking_status = ['smokes', 'formerly smoked', 'never smoked']

col1 = ["#364F6B", "#FC5185"]
colormap1 = LinearSegmentedColormap.from_list(
    "", col1, N=256)
col2 = ["#364F6B", "#3FC1C9"]
colormap2 = LinearSegmentedColormap.from_list("", col2)

stroke = pd.crosstab(unhealthy_person['gender'], [
                     unhealthy_person['smoking_status']], normalize='index').loc[gender, smoking_status]
no_stroke = pd.crosstab(healthy_person['gender'], [
                        healthy_person['smoking_status']], normalize='index').loc[gender, smoking_status]

sns.heatmap(ax=ax0, data=stroke, linewidths=0,
            square=True, cbar_kws={"orientation": "horizontal"}, cbar=False, linewidth=3, cmap=col1, annot=True, fmt='1.0%', annot_kws={"fontsize": 14}, alpha=0.9)

sns.heatmap(ax=ax1, data=no_stroke, linewidths=0,
            square=True, cbar_kws={"orientation": "horizontal"}, cbar=False, linewidth=3, cmap=col2, annot=True, fmt='1.0%', annot_kws={"fontsize": 14}, alpha=0.9)


ax0.text(0, -1., 'Distribution of Strokes with Gender & Smoking Status',
         {'font': 'Serif', 'color': 'black', 'weight': 'bold', 'size': 25})

ax0.text(0, -0.1, 'Stroke Pecentage ',
         {'font': 'serif', 'color': "#FC5185", 'size': 20}, alpha=0.9)
ax1.text(0, -0.1, 'No Stroke Percentage',
         {'font': 'serif', 'color': "#364F6B", 'size': 20}, alpha=0.9)

ax0.set_xticklabels(['Smoke', 'Quit', 'Never'])
ax1.set_xticklabels(['Smoke', 'Quit', 'Never'])

ax0.set_xlabel('')
ax0.set_ylabel('')
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.axes.get_yaxis().set_visible(False)
plt.show()