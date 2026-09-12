"""导出 MultiHeadRelationalModule 为 ONNX,供 Netron 可视化。

用法:
    python export_relmod.py
生成: relmod.onnx —— 拖入 https://netron.app 即可查看。
"""
import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadRelationalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ch = 16
        self.conv2_ch = 20
        self.node_size = 64
        self.out_dim = 5
        self.ch_in = 3
        self.sp_coord_dim = 2
        self.N = 7 * 7          # tokens = cH * cW
        self.n_heads = 3

        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=1)

        proj_in = self.conv2_ch + self.sp_coord_dim
        proj_out = self.n_heads * self.node_size
        self.k_proj = nn.Linear(proj_in, proj_out)
        self.q_proj = nn.Linear(proj_in, proj_out)
        self.v_proj = nn.Linear(proj_in, proj_out)

        self.k_lin = nn.Linear(self.node_size, self.N)
        self.q_lin = nn.Linear(self.node_size, self.N)
        self.a_lin = nn.Linear(self.N, self.N)

        node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        N, _, _, _ = x.shape
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        _, _, cH, cW = x.shape
        xcoords = torch.arange(cW, device=x.device).repeat(cH, 1).float() / cW
        ycoords = torch.arange(cH, device=x.device).repeat(cW, 1).transpose(1, 0).float() / cH
        coords = torch.stack([xcoords, ycoords], dim=0).unsqueeze(0).repeat(N, 1, 1, 1)

        x = torch.cat([x, coords], dim=1)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)

        K = self.k_norm(rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=self.n_heads))
        Q = self.q_norm(rearrange(self.q_proj(x), "b n (h d) -> b h n d", h=self.n_heads))
        V = self.v_norm(rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=self.n_heads))

        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K))
        A = torch.nn.functional.softmax(self.a_lin(A), dim=3)

        E = torch.einsum("bhfc,bhcd->bhfd", A, V)
        E = rearrange(E, "b h n d -> b n (h d)")
        E = self.norm1(torch.relu(self.linear1(E)))
        E = E.max(dim=1)[0]
        return torch.nn.functional.elu(self.linear2(E))


if __name__ == "__main__":
    model = MultiHeadRelationalModule().eval()
    dummy = torch.randn(1, 3, 7, 7)          # 输入必须是 7x7,才能满足 N=49
    out_path = "relmod.onnx"
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported -> {out_path}, output shape: {model(dummy).shape}")
