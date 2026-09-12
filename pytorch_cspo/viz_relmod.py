"""生成 MultiHeadRelationalModule 的多种可视化。

产物:
  relmod_torchview.png   —— torchview 自动展开的计算图
  relmod_summary.txt     —— torchinfo 的表格摘要
  relmod_schematic.png   —— 手绘的分块示意图 (graphviz)
"""
from export_relmod import MultiHeadRelationalModule
import torch
from torchview import draw_graph
from torchinfo import summary
import graphviz

model = MultiHeadRelationalModule().eval()
dummy_shape = (1, 3, 7, 7)

# 1) torchview: 自动追踪计算图
g = draw_graph(model, input_size=dummy_shape, expand_nested=True,
               graph_name="MultiHeadRelationalModule",
               save_graph=True, filename="relmod_torchview", directory=".")
print("torchview -> relmod_torchview.png")

# 2) torchinfo: 逐层表格
with open("relmod_summary.txt", "w") as f:
    f.write(str(summary(model, input_size=dummy_shape, depth=3, verbose=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size"))))
print("torchinfo -> relmod_summary.txt")

# 3) 手绘 schematic: 反映论文风格的分块结构
dot = graphviz.Digraph("relmod", format="png")
dot.attr(rankdir="TB", bgcolor="white", fontname="Helvetica")
dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")

def n(name, label, color):
    dot.node(name, label, fillcolor=color)

n("in",   "Input\n(B, 3, 7, 7)",                  "#e0f2fe")
n("c1",   "Conv1x1: 3→16\n+ ReLU",                "#fde68a")
n("c2",   "Conv1x1: 16→20\n+ ReLU",               "#fde68a")
n("cat",  "Concat spatial coords\n(B, 22, 7, 7)", "#fef3c7")
n("flat", "Flatten HW → tokens\n(B, 49, 22)",     "#fef3c7")

n("kp",   "k_proj\n(22→192)",  "#c7d2fe")
n("qp",   "q_proj\n(22→192)",  "#c7d2fe")
n("vp",   "v_proj\n(22→192)",  "#c7d2fe")

n("kh",   "reshape → (B,3,49,64)\nLayerNorm", "#e0e7ff")
n("qh",   "reshape → (B,3,49,64)\nLayerNorm", "#e0e7ff")
n("vh",   "reshape → (B,3,49,64)\nLayerNorm", "#e0e7ff")

n("qlin", "q_lin (64→49)", "#fbcfe8")
n("klin", "k_lin (64→49)", "#fbcfe8")
n("add",  "ELU(Q_lin + K_lin)\n(B,3,49,49)",       "#f9a8d4")
n("alin", "a_lin (49→49)\n+ Softmax(dim=3)",       "#f472b6")

n("ein",  "Einsum A·V\n(B,3,49,64)",  "#a7f3d0")
n("mg",   "merge heads → (B,49,192)", "#a7f3d0")
n("l1",   "Linear (192→64)\n+ ReLU + LayerNorm",  "#6ee7b7")
n("mx",   "max over tokens\n(B, 64)",  "#34d399")
n("l2",   "Linear (64→5)\n+ ELU",     "#fca5a5")
n("out",  "Q-values\n(B, 5)",         "#fecaca")

edges = [
    ("in","c1"), ("c1","c2"), ("c2","cat"), ("cat","flat"),
    ("flat","kp"), ("flat","qp"), ("flat","vp"),
    ("kp","kh"), ("qp","qh"), ("vp","vh"),
    ("qh","qlin"), ("kh","klin"),
    ("qlin","add"), ("klin","add"),
    ("add","alin"),
    ("alin","ein"), ("vh","ein"),
    ("ein","mg"), ("mg","l1"), ("l1","mx"), ("mx","l2"), ("l2","out"),
]
for a, b in edges:
    dot.edge(a, b)

dot.render("relmod_schematic", cleanup=True)
print("schematic -> relmod_schematic.png")
