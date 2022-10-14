#! Users/Kathy/anaconda3/envs/seaflow/bin/python3
## script that contains graphing helper functions

# libraries to import 
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt

## easily aestheticize plotly figures, especially facet ones
def pretty_plot(fig, rescale_x=False, rescale_y=False, x_label="", y_label=""):
    # rescale each x axis if yes
    if rescale_x:
        # rescale each axis to be different 
        for k in fig.layout:
            if re.search('xaxis[1-9]+', k):
                fig.layout[k].update(matches=None)
        
        # add x axis back in
        fig.update_xaxes(showticklabels=True, 
                      tickangle=0, tickfont=dict(size=10))
    # rescale each y axis if yes
    if rescale_y:
        # rescale each axis to be different 
        for k in fig.layout:
            if re.search('yaxis[1-9]+', k):
                fig.layout[k].update(matches=None)
        
        # add y axis back in
        fig.update_yaxes(showticklabels=True, col=2)

    # shorten default subplot titles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

#     # fix annotations to make them horizontal
#     for annotation in fig['layout']['annotations']: 
#         annotation['textangle']= 0

    # horizontal colorbar
    fig.update_layout(coloraxis_colorbar_x=-0.15)

    # #decrease font size 
    fig.update_annotations(font_size=10)

    # hide subplot y-axis titles and x-axis titles
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''
    # keep all other annotations and add single y-axis and x-axis title:
    fig.update_layout(
        # keep the original annotations and add a list of new annotations:
        annotations = list(fig.layout.annotations) + 
        [go.layout.Annotation(
                x=-0.07,
                y=0.5,
                font=dict(
                    size=16, color = 'black'
                ),
                showarrow=False,
                text=y_label,
                textangle=-90,
                xref="paper",
                yref="paper"
            )
        ] +
        [go.layout.Annotation(
                x=0.5,
                y=-0.1,
                font=dict(
                    size=16, color = 'black'
                ),
                showarrow=False,
                text=x_label,
                textangle=-0,
                xref="paper",
                yref="paper"
            )
        ]
    )
    return(fig)

## helper function for double y axes in matplotlib (still need to adjust to not be hardcoded)
def plt_double_axis(x, y1, y2, x_title=None, y1_title=None, y2_title=None):
    fig, axs = plt.subplots(figsize=(20,8))

    ax2 = axs.twinx()
    axs.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')

    axs.set_xlabel(x_title)
    axs.set_ylabel(y1_title, color='g')
    ax2.set_ylabel(y2_title, color='b')

    # returns output of figure
    return(fig)
