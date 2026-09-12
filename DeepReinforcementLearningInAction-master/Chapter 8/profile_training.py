"""
性能剖析脚本：找出训练循环里最慢的部分。

用法：
    python profile_training.py

脚本会构造和 notebook 里一样的环境、网络与训练步骤，然后运行 N 轮真实训练，
分阶段计时并输出耗时占比。目的是在优化之前确认瓶颈在哪一层（环境步进 /
帧预处理 / Q 前向 / 训练反向传播），而不是凭猜测改代码。

评估阶段：
    env_step       : env.step 六次（action_repeats=6）
    preprocess     : downscale_obs + torch tensor + .to(device) 的总时间
    q_forward      : 单帧 Qmodel(state1) 用于选动作
    train_step     : minibatch_train + loss.backward + opt.step
    other          : Python 控制流、replay.add_memory、结果拷贝等杂项

依赖：与 Ch8_book_gpu.ipynb 相同。
"""

import time
from collections import deque
from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch import nn, optim

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


# =============== 环境 ===============
env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="rgb_array")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")


# =============== 预处理（与 notebook 一致） ===============
def downscale_obs(obs, new_size=(42, 42), to_gray=True):
    # cv2.resize 比 skimage 快数倍；INTER_AREA 是缩小图像的推荐算法，
    # 对 42×42 这种尺度也不需要 anti_aliasing。
    small = cv2.resize(obs, new_size, interpolation=cv2.INTER_AREA)
    if to_gray:
        return small.max(axis=2)
    return small


def prepare_state(state):
    return (torch.from_numpy(downscale_obs(state, to_gray=True))
            .float().unsqueeze(dim=0).to(device))


def prepare_initial_state(state, N=3):
    t = torch.from_numpy(downscale_obs(state, to_gray=True)).float()
    return t.repeat((N, 1, 1)).unsqueeze(dim=0).to(device)


# =============== 网络（与 notebook 一致） ===============
class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        return y.flatten(start_dim=1)


class Gnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(576, 256)
        self.linear2 = nn.Linear(256, 12)

    def forward(self, s1, s2):
        return self.linear2(F.relu(self.linear1(torch.cat((s1, s2), dim=1))))


class Fnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(300, 256)
        self.linear2 = nn.Linear(256, 288)

    def forward(self, state, action):
        act = torch.zeros(action.shape[0], 12, device=action.device)
        idx = torch.stack((torch.arange(action.shape[0], device=action.device),
                           action.squeeze()), dim=0)
        act[tuple(idx)] = 1.
        return self.linear2(F.relu(self.linear1(torch.cat((state, act), dim=1))))


class Qnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear1 = nn.Linear(288, 100)
        self.linear2 = nn.Linear(100, 12)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim=2).view(y.shape[0], -1, 32).flatten(start_dim=1)
        return self.linear2(F.elu(self.linear1(y)))


# =============== 参数、模型、回放缓冲 ===============
params = {
    'batch_size': 256, 'beta': 0.20, 'lambda': 0.1, 'eta': 0.6,
    'gamma': 0.8, 'action_repeats': 6, 'frames_per_state': 3,
}

Qmodel = Qnetwork().to(device)
encoder = Phi().to(device)
forward_model = Fnet().to(device)
inverse_model = Gnet().to(device)

forward_loss_fn = nn.MSELoss(reduction='none')
inverse_loss_fn = nn.CrossEntropyLoss(reduction='none')
qloss_fn = nn.MSELoss()

all_params = (list(Qmodel.parameters()) + list(encoder.parameters())
              + list(forward_model.parameters()) + list(inverse_model.parameters()))
opt = optim.Adam(lr=0.001, params=all_params)


class Replay:
    def __init__(self, N=1000, batch_size=256):
        self.N = N
        self.batch_size = batch_size
        self.memory = []

    def add(self, s1, a, r, s2):
        if len(self.memory) < self.N:
            self.memory.append((s1, a, r, s2))
        else:
            self.memory[np.random.randint(0, self.N - 1)] = (s1, a, r, s2)

    def get_batch(self):
        idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        b = [self.memory[i] for i in idx]
        s1 = torch.stack([x[0].squeeze(0) for x in b]).to(device)
        a = torch.tensor([x[1] for x in b]).long().to(device)
        r = torch.tensor([x[2] for x in b]).float().to(device)
        s2 = torch.stack([x[3].squeeze(0) for x in b]).to(device)
        return s1, a, r, s2


replay = Replay(N=1000, batch_size=params['batch_size'])


def ICM(s1, a, s2, forward_scale=1., inverse_scale=1e4):
    h1, h2 = encoder(s1), encoder(s2)
    h2_pred = forward_model(h1.detach(), a.detach())
    fe = forward_scale * forward_loss_fn(h2_pred, h2.detach()).sum(dim=1).unsqueeze(1)
    ap = inverse_model(h1, h2)
    ie = inverse_scale * inverse_loss_fn(ap, a.detach().flatten()).unsqueeze(1)
    return fe, ie


def minibatch_train():
    s1, a, r, s2 = replay.get_batch()
    a = a.view(-1, 1)
    r = r.view(-1, 1)
    fe, ie = ICM(s1, a, s2)
    reward = (1. / params['eta']) * fe.detach() + r
    qvals = Qmodel(s2)
    reward = reward + params['gamma'] * torch.max(qvals, dim=1)[0].unsqueeze(1)
    reward_pred = Qmodel(s1)
    reward_target = reward_pred.clone()
    idx = tuple(torch.stack((torch.arange(a.shape[0], device=device),
                             a.squeeze()), dim=0))
    reward_target[idx] = reward.squeeze()
    q_loss = 1e5 * qloss_fn(F.normalize(reward_pred),
                            F.normalize(reward_target.detach()))
    return fe, ie, q_loss


def loss_fn(q_loss, ie, fe):
    L = (1 - params['beta']) * ie + params['beta'] * fe
    L = L.sum() / L.flatten().shape[0]
    return L + params['lambda'] * q_loss


# =============== 预热：填满 replay ===============
print("\n预热：填充 replay buffer 到 batch_size...")
env.reset()
state1 = prepare_initial_state(env.render())
state_deque = deque(maxlen=params['frames_per_state'])
for _ in range(params['frames_per_state']):
    state_deque.append(prepare_state(env.render()))

while len(replay.memory) < params['batch_size']:
    action = np.random.randint(0, 12)
    for _ in range(params['action_repeats']):
        s2, _, done, trunc, _ = env.step(action)
        if done:
            env.reset()
            state1 = prepare_initial_state(env.render())
            for _ in range(params['frames_per_state']):
                state_deque.append(prepare_state(env.render()))
            break
        state_deque.append(prepare_state(s2))
    s2t = torch.stack(list(state_deque), dim=1)
    replay.add(state1, action, 0., s2t)
    if not done:
        state1 = s2t

# 让 GPU 也预热一次（避免第一次 CUDA 编译时间污染测量）
for _ in range(5):
    fe, ie, ql = minibatch_train()
    (loss_fn(ql, ie, fe)).backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()


# =============== 剖析主循环 ===============
N_ITERS = 500
timings = {k: 0.0 for k in
           ['env_step', 'preprocess', 'q_forward', 'train_step', 'other',
            'total']}
print(f"\n开始剖析：{N_ITERS} 轮真实训练迭代...\n")

env.reset()
state1 = prepare_initial_state(env.render())
state_deque = deque(maxlen=params['frames_per_state'])
for _ in range(params['frames_per_state']):
    state_deque.append(prepare_state(env.render()))

for i in range(N_ITERS):
    t_iter = time.perf_counter()

    # --- Q 前向：选动作 ---
    t0 = time.perf_counter()
    q_val_pred = Qmodel(state1)
    action = int(torch.argmax(q_val_pred))
    torch.cuda.synchronize()
    timings['q_forward'] += time.perf_counter() - t0

    # --- 环境步进 + 帧预处理：分开计时 ---
    env_dt = 0.0
    prep_dt = 0.0
    for _ in range(params['action_repeats']):
        t0 = time.perf_counter()
        s2, _, done, trunc, _ = env.step(action)
        env_dt += time.perf_counter() - t0

        if done:
            t0 = time.perf_counter()
            env.reset()
            state1 = prepare_initial_state(env.render())
            for _ in range(params['frames_per_state']):
                state_deque.append(prepare_state(env.render()))
            prep_dt += time.perf_counter() - t0
            break

        t0 = time.perf_counter()
        state_deque.append(prepare_state(s2))
        prep_dt += time.perf_counter() - t0
    timings['env_step'] += env_dt
    timings['preprocess'] += prep_dt

    # --- 其他：拼张量、写 replay ---
    t0 = time.perf_counter()
    s2t = torch.stack(list(state_deque), dim=1)
    replay.add(state1, action, 0., s2t)
    if not done:
        state1 = s2t
    timings['other'] += time.perf_counter() - t0

    # --- 训练一步 ---
    t0 = time.perf_counter()
    opt.zero_grad()
    fe, ie, ql = minibatch_train()
    loss = loss_fn(ql, ie, fe)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    timings['train_step'] += time.perf_counter() - t0

    timings['total'] += time.perf_counter() - t_iter


# =============== 结果 ===============
total = timings['total']
per_iter_ms = total * 1000 / N_ITERS
print(f"总迭代数: {N_ITERS}")
print(f"总耗时:   {total:.2f}s")
print(f"每轮平均: {per_iter_ms:.2f} ms  ({N_ITERS/total:.1f} it/s)")
print(f"预计训练 20000 步:  {20000 * per_iter_ms / 1000:.0f}s\n")

print(f"{'阶段':<12} {'ms/iter':>10} {'占比':>8}")
print("-" * 34)
for k in ['env_step', 'preprocess', 'q_forward', 'train_step', 'other']:
    v = timings[k]
    print(f"{k:<12} {v*1000/N_ITERS:>10.2f} {v/total*100:>7.1f}%")

# 简单结论
print("\n结论提示：")
env_pct = timings['env_step'] / total
prep_pct = timings['preprocess'] / total
train_pct = timings['train_step'] / total
gpu_pct = (timings['q_forward'] + timings['train_step']) / total
print(f"  GPU 相关（Q 前向 + 训练）占比: {gpu_pct*100:.1f}%")
print(f"  CPU 相关（环境 + 预处理）占比: {(env_pct+prep_pct)*100:.1f}%")
if env_pct > 0.4:
    print("  → 环境步进是最大瓶颈。优化路径：并行环境（AsyncVectorEnv）。")
if prep_pct > 0.15:
    print("  → 帧预处理占比明显。优化路径：cv2.resize / GPU 端 F.interpolate / "
          "去掉 anti_aliasing。")
if gpu_pct < 0.2:
    print("  → GPU 长时间空转。加大 batch_size 收益低；先解决 CPU 瓶颈。")
