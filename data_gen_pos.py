import json
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

GRID = (5, 5)

position_transformer = PositionSetTransformer(splits=GRID, element_ndim=3)


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

high_level_elements = {
    # position
    0: Eps_set,
    # color
    1: Ecs_set
}

high_level_elements_list = [Eps_set, Ecs_set]

evaluator = Evaluator(operations, expressionOperations, high_level_elements=high_level_elements)

hists_data = "hists_pos.pkcl"
images_data = "images_pos.pkcl"

images = list()
hists = list()

if os.path.exists(hists_data) and os.path.exists(images_data):
    with open(hists_data, "rb") as f:
        hists = pickle.load(f)
    with open(images_data, "rb") as f:
        images = pickle.load(f)

TOP_N = 30

E1 = E("top", "green")
E2 = E("right", "red")
E3 = E("any", "green")
E4 = E("any", "red")


query = E1 & E2
query_big = E3 + E4

sample_image = image_generator.generate(
    shape=(100, 100),
    steps=(10, 10),
    normal_element_ids={"e33", "e34"},
    random_state=1)

position_image = position_transformer.fit_transform(X=sample_image)
color_image = color_transformer.transform(sample_image)
sample_hist = create_histogram((position_image, color_image))

pos_data = {}
# len(hists)//100 + 1
for i in range(1, len(hists)//200 + 1):
    step = i * 200
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


    pos_data[step] = {
        "time_filling_py": time_filling_py,
        "time_filling_cpp": time_filling_cpp,
        "time_query_py": time_query_py,
        "time_query_cpp": time_query_cpp,
        "time_query_big_py": time_query_big_py,
        "time_query_big_cpp": time_query_big_cpp,
        "time_hist_py": time_hist_py,
        "time_hist_cpp": time_hist_cpp
    }

with open("pos_data.json", "w") as file:
    json.dump(pos_data, file, indent=4)
