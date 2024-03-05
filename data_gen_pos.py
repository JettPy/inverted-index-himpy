import json
import os
import pickle
import time
from himpy.executor import Parser, Evaluator
from himpy.histogram import operations, expressionOperations
from himpy.utils import E
from utils.datasets import ColorImageGenerator
from utils.feature_extraction import ColorSetTransformer, PositionSetTransformer, create_histogram
from utils.search_engine import SearchEngine

# =============================================================================================================

image_generator = ColorImageGenerator()
color_transformer = ColorSetTransformer()
parser = Parser()


# Ec_green        = E("e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16+e17+e18+e19+e20")
# Ec_yellow_green = E("e2+e3+e21+e22+e23+e24+e25+e26+e27+e28+e29+e30")
# Ec_red          = E("e31+e32+e33+e34+e35+e36+e37+e38+e39+e40")
# Ec_rose         = E("e32+e35+e36+e39+e40")
Ec_green        = E("e1+e2")
Ec_yellow_green = E("e2+e3")
Ec_red          = E("e31+e32")
Ec_rose         = E("e32")
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
E2 = E("center", "yellow_green")
E3 = E("top", "green")
E4 = E("any", "red")
E5 = E("left", "rose")


query = E1 * E2
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
    # test_data = hists

    time_filling_ds = []
    time_filling_py = []
    time_filling_cpp = []
    time_filling_prl = []
    time_query_ds = []
    time_query_py = []
    time_query_cpp = []
    time_query_prl = []
    time_query_big_ds = []
    time_query_big_py = []
    time_query_big_cpp = []
    time_query_big_prl = []
    time_hist_ds = []
    time_hist_py = []
    time_hist_cpp = []
    time_hist_prl = []

    for j in range(10):
        print(f"step: {i}, test: {j}")

        # print("filling ds")
        start_time_filling_ds = time.time()
        search_engine = SearchEngine(test_data, parser, evaluator)
        end_time_filling_ds = time.time()

        # print("filling py")
        start_time_filling_py = time.time()
        search_engine_py = SearchEngine(test_data, parser, evaluator, mode="classic")
        end_time_filling_py = time.time()

        # print("filling cpp")
        start_time_filling_cpp = time.time()
        search_engine_cpp = SearchEngine(test_data, parser, evaluator, rules=high_level_elements_list, mode="dll")
        end_time_filling_cpp = time.time()

        # print("filling prl")
        start_time_filling_prl = time.time()
        search_engine_prl = SearchEngine(test_data, parser, evaluator, mode="parallel")
        end_time_filling_prl = time.time()

        # print("query ds")
        start_time_query_ds = time.time()
        ranked_images = search_engine.retrieve(query, top_n=TOP_N)
        end_time_query_ds = time.time()

        # print("query py")
        start_time_query_py = time.time()
        ranked_images_py = search_engine_py.retrieve(query, top_n=TOP_N)
        end_time_query_py = time.time()

        # print("query cpp")
        start_time_query_cpp = time.time()
        ranked_images_cpp = search_engine_cpp.retrieve(query, top_n=TOP_N)
        end_time_query_cpp = time.time()

        # print("query prl")
        start_time_query_prl = time.time()
        ranked_images_prl = search_engine_prl.retrieve(query, top_n=TOP_N)
        end_time_query_prl = time.time()

        # print("query big ds")
        start_time_query_big_ds = time.time()
        ranked_images_big = search_engine.retrieve(query_big, top_n=TOP_N)
        end_time_query_big_ds = time.time()

        # print("query big py")
        start_time_query_big_py = time.time()
        ranked_images_big_py = search_engine_py.retrieve(query_big, top_n=TOP_N)
        end_time_query_big_py = time.time()

        # print("query big cpp")
        start_time_query_big_cpp = time.time()
        ranked_images_big_cpp = search_engine_cpp.retrieve(query_big, top_n=TOP_N)
        end_time_query_big_cpp = time.time()

        # print("query big prl")
        start_time_query_big_prl = time.time()
        ranked_images_big_prl = search_engine_prl.retrieve(query_big, top_n=TOP_N)
        end_time_query_big_prl = time.time()

        # print("hist ds")
        start_time_hist_ds = time.time()
        ranked_images_hist = search_engine.retrieve(sample_hist, top_n=TOP_N)
        end_time_hist_ds = time.time()

        # print("hist py")
        start_time_hist_py = time.time()
        ranked_images_hist_py = search_engine_py.retrieve(sample_hist, top_n=TOP_N)
        end_time_hist_py = time.time()

        # print("hist cpp")
        start_time_histy_cpp = time.time()
        ranked_images_hist_cpp = search_engine_cpp.retrieve(sample_hist, top_n=TOP_N)
        end_time_hist_cpp = time.time()

        # print("hist prl")
        start_time_histy_prl = time.time()
        ranked_images_hist_prl = search_engine_prl.retrieve(sample_hist, top_n=TOP_N)
        end_time_hist_prl = time.time()

        time_filling_ds.append((end_time_filling_ds - start_time_filling_ds) * 1000)
        time_filling_py.append((end_time_filling_py - start_time_filling_py) * 1000)
        time_filling_cpp.append((end_time_filling_cpp - start_time_filling_cpp) * 1000)
        time_filling_prl.append((end_time_filling_prl - start_time_filling_prl) * 1000)
        time_query_ds.append((end_time_query_ds - start_time_query_ds) * 1000)
        time_query_py.append((end_time_query_py - start_time_query_py) * 1000)
        time_query_cpp.append((end_time_query_cpp - start_time_query_cpp) * 1000)
        time_query_prl.append((end_time_query_prl - start_time_query_prl) * 1000)
        time_query_big_ds.append((end_time_query_big_ds - start_time_query_big_ds) * 1000)
        time_query_big_py.append((end_time_query_big_py - start_time_query_big_py) * 1000)
        time_query_big_cpp.append((end_time_query_big_cpp - start_time_query_big_cpp) * 1000)
        time_query_big_prl.append((end_time_query_big_prl - start_time_query_big_prl) * 1000)
        time_hist_ds.append((end_time_hist_ds - start_time_hist_ds) * 1000)
        time_hist_py.append((end_time_hist_py - start_time_hist_py) * 1000)
        time_hist_cpp.append((end_time_hist_cpp - start_time_histy_cpp) * 1000)
        time_hist_prl.append((end_time_hist_prl - start_time_histy_prl) * 1000)

    pos_data[step] = {
        "time_filling_ds": time_filling_ds,
        "time_filling_py": time_filling_py,
        "time_filling_cpp": time_filling_cpp,
        "time_filling_prl": time_filling_prl,
        "time_query_ds": time_query_ds,
        "time_query_py": time_query_py,
        "time_query_cpp": time_query_cpp,
        "time_query_prl": time_query_prl,
        "time_query_big_ds": time_query_big_ds,
        "time_query_big_py": time_query_big_py,
        "time_query_big_cpp": time_query_big_cpp,
        "time_query_big_prl": time_query_big_prl,
        "time_hist_ds": time_hist_ds,
        "time_hist_py": time_hist_py,
        "time_hist_cpp": time_hist_cpp,
        "time_hist_prl": time_hist_prl
    }

    with open("pos_data.json", "w") as file:
        json.dump(pos_data, file, indent=4)

#
# with open("pos_data.json", "r") as file:
#     pos_data = json.load(file)
#
# pos_data_1 = {}
#
# # len(hists)//100 + 1
# for i in range(1, len(hists)//200 + 1):
#     step = i * 200
#     test_data = hists[:step]
#
#     time_filling_ds = []
#     time_query_ds = []
#     time_query_big_ds = []
#     time_hist_ds = []
#
#
#     for j in range(10):
#         print(f"step: {i}, test: {j}")
#
#         start_time_filling_ds = time.time()
#         search_engine = SearchEngine(test_data, parser, evaluator, use_index=False, use_cpp=False)
#         end_time_filling_ds = time.time()
#
#         start_time_query_ds = time.time()
#         ranked_images = search_engine.retrieve(query, top_n=TOP_N)
#         end_time_query_ds = time.time()
#
#         start_time_query_big_ds = time.time()
#         ranked_images_big = search_engine.retrieve(query_big, top_n=TOP_N)
#         end_time_query_big_ds = time.time()
#
#         start_time_hist_ds = time.time()
#         ranked_images_hist = search_engine.retrieve(sample_hist, top_n=TOP_N)
#         end_time_hist_ds = time.time()
#
#         time_filling_ds.append((end_time_filling_ds - start_time_filling_ds) * 1000)
#         time_query_ds.append((end_time_query_ds - start_time_query_ds) * 1000)
#         time_query_big_ds.append((end_time_query_big_ds - start_time_query_big_ds) * 1000)
#         time_hist_ds.append((end_time_hist_ds - start_time_hist_ds) * 1000)
#
#
#     a = {
#         "time_filling_ds": time_filling_ds,
#         "time_query_ds": time_query_ds,
#         "time_query_big_ds": time_query_big_ds,
#         "time_hist_ds": time_hist_ds
#     }
#
#     pos_data_1[step] = dict(list(pos_data[str(step)].items()) + list(a.items()))
#
