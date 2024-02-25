import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as image_utils
from matplotlib.patches import Rectangle

from ..feature_extraction.base import extract_element_set, filter_data


"""
Position Grid
"""


def plot_position_grid(position_transformer, ax, element_ids=None):
    for el in position_transformer.find_elements(element_ids):
        bottom, height = el[1][0], el[1][2]-el[1][0]
        left, width = el[1][1], el[1][3]-el[1][1]
        right = left + width
        top = bottom + height
        ax.add_patch(Rectangle(xy=(left, bottom), width=width, height=height, fill=False,
                                      label=el, edgecolor="red", linewidth=2))
        ax.text(0.5*(left+right), 0.5*(bottom+top), el[0],
                horizontalalignment="center", verticalalignment="center", fontsize=15, color="red")
    return ax


"""
Objects
"""


def plot_object_ids(object_image, ax, element_ids=None):
    if element_ids is None:
        element_ids = np.unique(object_image[object_image != 0])
    for element_id in element_ids:
        indices = np.where(object_image==int(element_id))
        if len(indices) == 2 and len(indices[0]) > 0:
            center = indices[1].mean(), indices[0].mean()
            ax.text(*center, str(element_id),
                    horizontalalignment="center", verticalalignment="center", fontsize=15, color="yellow")

    return ax


"""
Histogram Operations
"""


# TODO: replace with show_operation_result
def show_operation_result_(I, merged_image, HE1, HE2, HE3, transformers, titles=("E1", "E2", "Result")):
    import matplotlib.pyplot as plt

    E1_set = extract_element_set(HE1, 2)
    E2_set = extract_element_set(HE2, 2)
    E3_set = extract_element_set(HE3, 2)

    E1_image = filter_data(I, merged_image, HE1.elements())
    E2_image = filter_data(I, merged_image, HE2.elements())
    E3_image = filter_data(I, merged_image, HE3.elements())

    fig, axes = plt.subplots(1, 3, figsize=(14, 20))
    axes[0].set_title(titles[0])
    axes[0].imshow(E1_image)
    axes[0] = plot_position_grid(transformers[0], axes[0], E1_set[0])
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].set_title(titles[1])
    axes[1].imshow(E2_image)
    axes[1] = plot_position_grid(transformers[0], axes[1], E2_set[0])
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[2].set_title(titles[2])
    axes[2].imshow(E3_image)
    axes[2] = plot_position_grid(transformers[0], axes[2], E3_set[0])
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)

    plt.show()


def show_operation_result(I, merged_image, obj_index, HE1, HE2, HE3, transformers, titles=("E1", "E2", "Result")):
    import matplotlib.pyplot as plt

    E1_set = extract_element_set(HE1, 2)
    E2_set = extract_element_set(HE2, 2)
    E3_set = extract_element_set(HE3, 2)

    E1_image = filter_data(I, merged_image, HE1.elements())
    E2_image = filter_data(I, merged_image, HE2.elements())
    E3_image = filter_data(I, merged_image, HE3.elements())

    fig, axes = plt.subplots(1, 3, figsize=(14, 20))
    axes[0].set_title(titles[0])
    axes[0].imshow(I)
    axes[0].imshow(E1_image, alpha=0.8)
    axes[0] = plot_position_grid(transformers[0], axes[0], E1_set[0])
    axes[0] = plot_object_ids(merged_image[obj_index], axes[0], E1_set[1])
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].set_title(titles[1])
    axes[1].imshow(I)
    axes[1].imshow(E2_image, alpha=0.8)
    axes[1] = plot_position_grid(transformers[0], axes[1], E2_set[0])
    axes[1] = plot_object_ids(merged_image[obj_index], axes[1], E2_set[1])
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[2].set_title(titles[2])
    axes[2].imshow(I)
    axes[2].imshow(E3_image, alpha=0.8)
    axes[2] = plot_position_grid(transformers[0], axes[2], E3_set[0])
    axes[2] = plot_object_ids(merged_image[obj_index], axes[2], E3_set[1])
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)

    plt.show()


"""
Image Retrieval
"""


def show_retrieved_images(ranked_images, images, title=None, limit=11, cols=5):

    img_limit = len(ranked_images) if limit > len(ranked_images) else limit
    #     if limit and img_limit > limit:
    #         img_limit = limit
    row_num = -(-img_limit // cols)

    # show title
    fig, ax = plt.subplots(1, figsize=(15, 1))
    ax.text(0.5, 0.5, title, clip_on=True, ha="center", va="center", fontsize=16, weight="bold")
    ax.axis("off")
    plt.show()

    # show images
    fig, axs = plt.subplots(row_num, cols, figsize=(15, 4 * row_num), squeeze=False)
    for i in range(row_num):
        for j in range(cols):
            indx = i * cols + j
            if indx >= img_limit:
                fig.delaxes(axs[i, j])
            else:
                I = None
                if isinstance(images[0], str):
                    # ranked images and their paths
                    I = image_utils.imread(images[indx])
                elif isinstance(images[0], np.ndarray):
                    # all images and pick id from ranked images
                    I = images[ranked_images[indx][0]]
                else:
                    raise ValueError
                axs[i, j].imshow(I)
                axs[i, j].set_title(
                    "rank={}\nid={}\nscore={:0.4f}".format(
                        indx + 1,
                        ranked_images[indx][0],
                        ranked_images[indx][1]))
                axs[i, j].axis("off")
    plt.tight_layout()
    plt.show()
