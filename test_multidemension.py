import os
import pickle
import time
from himpy.executor import Parser, Evaluator
from himpy.histogram import operations, expressionOperations
from himpy.utils import E
from utils.datasets import ColorImageGenerator
from utils.feature_extraction import ColorSetTransformer, PositionSetTransformer, create_histogram
from utils.search_engine import SearchEngine, InvertedIndexCpp

# =============================================================================================================

image_generator = ColorImageGenerator()
color_transformer = ColorSetTransformer()
parser = Parser()

# =============================================================================================================

# Definition of high-level positional elements

Ec_green        = E("e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16+e17+e18+e19+e20")
Ec_yellow_green = E("e2+e3+e21+e22+e23+e24+e25+e26+e27+e28+e29+e30")
Ec_red          = E("e31+e32+e33+e34+e35+e36+e37+e38+e39+e40")
Ec_rose         = E("e32+e35+e36+e39+e40")
Ec_any          = E("e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16+e17+e18+e19+e20+e21+e22+e23+e24+e25+e26+e27+e28+e29+e30+e31+e32+e33+e34+e35+e36+e37+e38+e39+e40")

Ecs = [
    ("green", Ec_green),
    ("yellow_green", Ec_yellow_green),
    ("red", Ec_red),
    ("rose", Ec_rose),
    ("any", Ec_any)
]

Ecs_set = { name: parser.parse_set(Ec.value) for name, Ec in Ecs}

# =============================================================================================================

# Grid params: 5 splits along Y, and 5 along X
GRID = (5, 5)

# Create a position transformer
position_transformer = PositionSetTransformer(splits=GRID, element_ndim=3)

# =============================================================================================================

# Definition of high-level positional elements

Ep_top    = E("1+2+3+4+5+6+7+8+9+10")
Ep_bottom = E("16+17+18+19+20+21+22+23+24+25")
Ep_left   = E("1+2+6+7+11+12+16+17+21+22")
Ep_right  = E("4+5+9+10+14+15+19+20+24+25")
Ep_center = E("7+8+9+12+13+14+17+18+19")
Ep_any = E("1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25")

Eps = [
    ("top", Ep_top),
    ("bottom", Ep_bottom),
    ("left", Ep_left),
    ("right", Ep_right),
    ("center", Ep_center),
    ("any", Ep_any)
]


Eps_set = { name: parser.parse_set(Ep.value) for name, Ep in Eps}

# =============================================================================================================

high_level_elements = {
    # position
    0: Eps_set,
    # color
    1: Ecs_set
}

high_level_elements_list = [Eps_set, Ecs_set]

# =============================================================================================================

# Initialize an evaluator
evaluator = Evaluator(operations, expressionOperations, high_level_elements=high_level_elements)

# =============================================================================================================

# Images with normal distrubited some elements
images = [
    image_generator.generate(
        shape=(100, 100),
        steps=(10, 10),
        normal_element_ids={"e1", "e10", "e11", "e12", "e31", "e32", "e33", "e34"},
        random_state=i+100)
    for i in range(5000)
]

# Images with uniform distributed elements
images += [
    image_generator.generate(
        shape=(100, 100),
        steps=(10, 10),
        random_state=i+100)
    for i in range(5000)
]

# =============================================================================================================

hists_data = "hists_pos.pkcl"
images_data = "images_pos.pkcl"

# Create histograms for the images
hists = list()
limit = len(images)

position_image = position_transformer.fit_transform(X=images[0], y=None)
if not os.path.exists(hists_data) or not os.path.exists(images_data):
    for indx, image in enumerate(images):
        color_image = color_transformer.fit_transform(X=image, y=None)
        hist = create_histogram((position_image, color_image))
        hists.append((indx, hist))
        print("\rCurrent image index: {}/{}".format(indx + 1, limit), end="")
    print()

# =============================================================================================================


if os.path.exists(hists_data) and os.path.exists(images_data):
    with open(hists_data, "rb") as f:
        hists = pickle.load(f)
    with open(images_data, "rb") as f:
        images = pickle.load(f)
else:
    with open(hists_data, "wb") as f:
        pickle.dump(hists, f)
    with open(images_data, "wb") as f:
        pickle.dump(images, f)

# =============================================================================================================

# Initialize a search engine
search_engine_default = SearchEngine(hists, parser, evaluator)
search_engine = SearchEngine(hists, parser, evaluator, mode="classic")
search_engine_cpp = SearchEngine(hists, parser, evaluator, rules=high_level_elements_list, mode="dll")

# =============================================================================================================

TOP_N = 30

# =============================================================================================================

# Elements
E1 = E("top", "green")
E2 = E("right", "red")

E3 = E("any", "any")
E4 = E("any", "any")

E5 = E("any", "green")
E6 = E("any", "any")

# Define your query
query = E5 + E2
# query = E1 + E2

start_time = time.time()
ranked_images = search_engine.retrieve(query, top_n=TOP_N)
end_time = time.time()
execution_time = (end_time - start_time) * 1000
print("Execution time (query py) in milliseconds:", execution_time)
print("Total retrieved images:", len(ranked_images))
# print(ranked_images)

# =============================================================================================================

start_time = time.time()
ranked_images = search_engine_cpp.retrieve(query, top_n=TOP_N)
end_time = time.time()
execution_time = (end_time - start_time) * 1000
print("Execution time (query cpp) in milliseconds:", execution_time)
print("Total retrieved images:", len(ranked_images))
# print(ranked_images)

# =============================================================================================================

# fig = show_rank_images(images, ranked_images, limit=TOP_N,
#                        title="Top {}: <b>{}</b>".format(TOP_N, query.value))
# fig.show()

# =============================================================================================================

# Generate a new sample image
sample_image = image_generator.generate(
    shape=(100, 100),
    steps=(10, 10),
    normal_element_ids={"e33", "e34"},
    random_state=1)

# =============================================================================================================

# Transform the image to histogram
position_image = position_transformer.fit_transform(X=sample_image)
color_image = color_transformer.transform(sample_image)
sample_hist = create_histogram((position_image, color_image))

# =============================================================================================================

# Retrieve images similar to the sample
start_time = time.time()
ranked_images__sample = search_engine.retrieve(sample_hist, top_n=TOP_N)
end_time = time.time()
execution_time = (end_time - start_time) * 1000
print("Execution time (hist py) in milliseconds:", execution_time)
print("Total retrieved images:", len(ranked_images__sample))
# print(ranked_images__sample)

# =============================================================================================================

start_time = time.time()
ranked_images__sample = search_engine_cpp.retrieve(sample_hist, top_n=TOP_N)
end_time = time.time()
execution_time = (end_time - start_time) * 1000
print("Execution time (hist cpp) in milliseconds:", execution_time)
print("Total retrieved images:", len(ranked_images__sample))
# print(ranked_images__sample)

# =============================================================================================================

# fig = make_subplots(rows=1, cols=1, subplot_titles=("Sample Image",))
#
# fig.add_trace(go.Image(z=sample_image, hoverinfo="skip"), row=1, col=1)
# fig.update_yaxes(showticklabels=False)
# fig.update_xaxes(showticklabels=False)
# fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), width=200, height=200)
# fig.show()
#
# fig = show_rank_images(images, ranked_images__sample,
#                        limit=TOP_N, title="Top {}: <b>Sample Image</b>".format(TOP_N))
# fig.show()

# =============================================================================================================

for i in range(len(hists)//100):
    step = i * 100
    test_data = hists[:step]
    search_engine = SearchEngine(hists, parser, evaluator, mode="classic")
    search_engine_cpp = SearchEngine(hists, parser, evaluator, mode="dll", rules=high_level_elements_list)
