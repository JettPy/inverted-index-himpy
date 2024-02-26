import json
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

high_level_elements_list = Ecs_set

evaluator = Evaluator(operations, expressionOperations, high_level_elements=high_level_elements_list)

hists_data = "hists_col.pkcl"
images_data = "images_col.pkcl"

images = list()
hists = list()

if os.path.exists(hists_data) and os.path.exists(images_data):
    with open(hists_data, "rb") as f:
        hists = pickle.load(f)
    with open(images_data, "rb") as f:
        images = pickle.load(f)

TOP_N = 30

E1 = E("green")
E2 = E("yellow_green")
E3 = E("any")

query = E1 * E2
query_big = E1 + E3


sample_image = image_generator.generate(
    shape=(100, 100),
    steps=(20, 20),
    random_state=1)

color_image = color_transformer.transform(X=sample_image)
sample_hist = create_histogram((color_image,))


col_data = {}
# len(hists)//100 + 1
for i in range(1, len(hists)//100 + 1):
    step = i * 100
    test_data = hists[:step]

    time_filling_py = []
    time_filling_cpp = []
    time_query_py = []
    time_query_cpp = []
    time_query_big_py = []
    time_query_big_cpp = []
    time_hist_py = []
    time_hist_cpp = []


    for j in range(10):
        print(f"step: {i}, test: {j}")

        start_time_filling_py = time.time()
        search_engine = SearchEngine(test_data, parser, evaluator, use_index=True)
        end_time_filling_py = time.time()

        start_time_filling_cpp = time.time()
        search_engine_cpp = InvertedIndexCpp(test_data, parser, high_level_elements_list)
        end_time_filling_cpp = time.time()

        start_time_query_py = time.time()
        ranked_images = search_engine.retrieve(query, top_n=TOP_N)
        end_time_query_py = time.time()

        start_time_query_cpp = time.time()
        ranked_images_cpp = search_engine_cpp.retrieve(query, top_n=TOP_N)
        end_time_query_cpp = time.time()

        start_time_query_big_py = time.time()
        ranked_images_big = search_engine.retrieve(query_big, top_n=TOP_N)
        end_time_query_big_py = time.time()

        start_time_query_big_cpp = time.time()
        ranked_images_big_cpp = search_engine_cpp.retrieve(query_big, top_n=TOP_N)
        end_time_query_big_cpp = time.time()

        start_time_hist_py = time.time()
        ranked_images_hist = search_engine.retrieve(sample_hist, top_n=TOP_N)
        end_time_hist_py = time.time()

        start_time_histy_cpp = time.time()
        ranked_images_hist_cpp = search_engine_cpp.retrieve(sample_hist, top_n=TOP_N)
        end_time_hist_cpp = time.time()

        time_filling_py.append((end_time_filling_py - start_time_filling_py) * 1000)
        time_filling_cpp.append((end_time_filling_cpp - start_time_filling_cpp) * 1000)
        time_query_py.append((end_time_query_py - start_time_query_py) * 1000)
        time_query_cpp.append((end_time_query_cpp - start_time_query_cpp) * 1000)
        time_query_big_py.append((end_time_query_big_py - start_time_query_big_py) * 1000)
        time_query_big_cpp.append((end_time_query_big_cpp - start_time_query_big_cpp) * 1000)
        time_hist_py.append((end_time_hist_py - start_time_hist_py) * 1000)
        time_hist_cpp.append((end_time_hist_cpp - start_time_histy_cpp) * 1000)


    col_data[step] = {
        "time_filling_py": time_filling_py,
        "time_filling_cpp": time_filling_cpp,
        "time_query_py": time_query_py,
        "time_query_cpp": time_query_cpp,
        "time_query_big_py": time_query_big_py,
        "time_query_big_cpp": time_query_big_cpp,
        "time_hist_py": time_hist_py,
        "time_hist_cpp": time_hist_cpp
    }

with open("col_data.json", "w") as file:
    json.dump(col_data, file, indent=4)
