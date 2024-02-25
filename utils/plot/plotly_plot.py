import numpy as np
from plotly.subplots import make_subplots

from ..feature_extraction.color import COLOR_ELEMENTS_RGB
from ..feature_extraction.base import convert_complete_histogram_values


"""
Color Elements
"""


def show_color_elements(elements=None, element_ids="all", element_type="hsb40", title="Low-level elements"):
    if element_type == "hsb40":
        x = np.array(tuple(COLOR_ELEMENTS_RGB.keys()))
        y = np.ones(len(COLOR_ELEMENTS_RGB))
        colors = ["rgb{}".format(rgb) for rgb in COLOR_ELEMENTS_RGB.values()]
        if element_ids != "all":
            # TODO: think to inverse np.any([x == el ...])
            y[np.all([x != el for el in tuple(element_ids)], axis=0)] = 0
        fig = make_subplots(rows=1, cols=1, subplot_titles=None)
        fig.add_bar(x=x, y=y, marker_color=colors, row=1, col=1)
        fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)
        fig.update_layout(margin=dict(l=60, r=60, t=60, b=60), plot_bgcolor="#fefefe", showlegend=False,
                          height=160 * 1, width=800, title_text=title)
        fig.show()
        return

    raise NotImplementedError


"""
Position Elements
"""


def plot_position_grid_plotly(position_transformer, fig, row, col, element_ids=None):
    max_x = position_transformer._size[1]
    max_y = position_transformer._size[0]
    for item_id, item in position_transformer.find_elements(element_ids):
        fig.add_shape(
            dict(type="rect",
                 x0=item[1], x1=item[3]+1, y0=max_y-item[0], y1=max_y-item[2]-1,
                 line_color="red",
                 line_width=2,
                 fillcolor="rgba(255,0,0,0)"),
            row=row, col=col,
        )
        fig.add_annotation(
            x=item[1]+(item[3]+1-item[1])*0.5, y=max_y-item[0]-(item[2]+1-item[0])*0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
            text=item_id,
            row=row, col=col)
    fig.update_yaxes(range=[0, max_y], constrain="domain", scaleanchor="x", row=row, col=col)
    fig.update_xaxes(range=[0, max_x], constrain="domain", scaleanchor="y", row=row, col=col)
    return fig


"""
Histogram
"""


def plot_histogram(elements, values, colors, fig, row, col):
    fig.add_bar(x=elements, y=values, marker_color=colors, width=0.5, row=row, col=col)
    fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=row, col=col)
    fig.update_yaxes(gridcolor="#bdbdbd", title="Counts", titlefont=dict(color="grey"), row=row, col=col)
    return fig


# Plot complete histogram view
def wrapper_show_complete_histogram():
    def _show_complete_histogram(H, title="Complete Histogram View"):
        complete_values = convert_complete_histogram_values(complete_elements, H)
        fig = make_subplots(rows=1, cols=1, subplot_titles=(title,))
        fig.add_bar(x=complete_elements,
                    y=complete_values,
                    marker_color=complete_colors, row=1, col=1)
        fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=1, col=1)
        fig.update_yaxes(gridcolor="#bdbdbd", title="Counts", titlefont=dict(color="grey"), row=1, col=1)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                          plot_bgcolor="#fefefe", showlegend=False, height=240, width=800)
        fig.show()

    complete_elements = [key for key, _ in sorted(COLOR_ELEMENTS_RGB.items(),
                                                  key=lambda x: int(x[0].lstrip("e")))]

    complete_colors = ["rgb{}".format(COLOR_ELEMENTS_RGB[item]) for item in complete_elements]
    return _show_complete_histogram


show_complete_histogram = wrapper_show_complete_histogram()


"""
Image Retrieval
"""


def show_rank_images(images, ranked_images, title, limit=11, cols=5):
    img_limit = len(ranked_images) if limit > len(ranked_images) else limit
    row_num = -(-img_limit // cols)

    fig = make_subplots(
        rows=row_num, cols=cols,
        subplot_titles=[
            "rank={}<br>id={}<br>score={:0.4f}".format(
                i + 1,
                image_id,
                image_score) for i, (image_id, image_score) in enumerate(ranked_images)
        ]
    )

    for i in range(row_num):
        for j in range(cols):
            indx = i * cols + j
            if indx >= img_limit:
                break
            else:
                fig.add_image(z=images[ranked_images[indx][0]], row=i + 1, col=j + 1)

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=100, b=20),
        plot_bgcolor="#fefefe",
        width=900, height=200 * row_num,
        showlegend=False,
        title={
            "text": title,
            "x": 0.5,
            "y": 1.0
        }
    )

    return fig