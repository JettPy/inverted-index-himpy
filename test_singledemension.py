import os
import pickle
import time
from himpy.executor import Parser, Evaluator
from himpy.histogram import operations, expressionOperations
from himpy.utils import E
from utils.datasets import ColorImageGenerator
from utils.feature_extraction import ColorSetTransformer, create_histogram
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

high_level_elements_list = Ecs_set

# =============================================================================================================

# Initialize an evaluator
evaluator = Evaluator(operations, expressionOperations, high_level_elements=high_level_elements_list)

# =============================================================================================================

# Images with normal distrubited some elements
images = [
    image_generator.generate(
        shape=(100, 100),
        steps=(20, 20),
        random_state=i+100)
    for i in range(10000)
]

# =============================================================================================================

hists_data = "hists_col.pkcl"
images_data = "images_col.pkcl"

# Create histograms for the images
hists = list()
limit = len(images)

if not os.path.exists(hists_data) or not os.path.exists(images_data):
    for indx, image in enumerate(images):
        color_image = color_transformer.fit_transform(X=image, y=None)
        hist = create_histogram((color_image,))
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
search_engine_default = SearchEngine(hists, parser, evaluator, use_index=False)
search_engine = SearchEngine(hists, parser, evaluator, use_index=True)
search_engine_cpp = InvertedIndexCpp(hists, parser, high_level_elements_list)

# =============================================================================================================

TOP_N = 30

# =============================================================================================================

# Elements
E1 = E("green")
E2 = E("yellow_green")
E3 = E("red")

# Define your query
query = E1 & E3
# query = E1 + E2

start_time = time.time()
ranked_images = search_engine.retrieve(query, top_n=TOP_N)
end_time = time.time()
execution_time = (end_time - start_time) * 1000
print("Execution time (query py) in milliseconds:", execution_time)
# print("Total retrieved images:", len(ranked_images))
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
    steps=(20, 20),
    random_state=1)

# =============================================================================================================

# Transform the image to histogram
color_image = color_transformer.transform(X=sample_image)
sample_hist = create_histogram((color_image,))

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
# fig = show_rank_images(images, ranked_images__sample1,
#                        limit=TOP_N, title="Top {}: <b>Sample Image</b>".format(TOP_N))
# fig.show()

# =============================================================================================================
