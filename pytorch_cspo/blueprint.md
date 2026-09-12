Yes. The best way is to separate your training program into **independent modules**, so you can change `resize`, state construction, action selection, environment stepping, replay, and training without touching everything else.

Here is the blueprint I would use for your current Mario RL code:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         CONFIG / PARAMS                              │
│                                                                     │
│  epochs                                                            │
│  frames_per_state                                                  │
│  action_repeats                                                    │
│  max_episode_len                                                  │
│  min_progress                                                      │
│  batch_size                                                        │
│  eps                                                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ENVIRONMENT MODULE                            │
│                                                                     │
│  env.reset()                                                       │
│  env.step(action)                                                  │
│  env.render()                                                      │
│                                                                     │
│  Input: action                                                     │
│  Output: raw RGB frame + reward + done + info                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OBSERVATION MODULE                              │
│                                                                     │
│  raw RGB frame                                                     │
│       │                                                             │
│       ▼                                                             │
│  downscale_obs()                                                   │
│       │                                                             │
│       ├── skimage.resize()     ← Version A                         │
│       │                                                             │
│       ├── cv2.resize()         ← Version B                         │
│       │                                                             │
│       └── torch resize()       ← Version C                         │
│       │                                                             │
│       ▼                                                             │
│  grayscale                                                         │
│       │                                                             │
│       ▼                                                             │
│  42 × 42                                                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       STATE MODULE                                  │
│                                                                     │
│  processed frame                                                   │
│       │                                                             │
│       ▼                                                             │
│  prepare_state()                                                   │
│       │                                                             │
│       ▼                                                             │
│  frame deque                                                       │
│       │                                                             │
│       │  [frame1, frame2, frame3, frame4]                          │
│       ▼                                                             │
│  torch.stack()                                                     │
│       │                                                             │
│       ▼                                                             │
│  STATE                                                               │
│                                                                     │
│  e.g. [batch, channels/frames, 42, 42]                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Q NETWORK                                  │
│                                                                     │
│                       state                                        │
│                         │                                           │
│                         ▼                                           │
│                      Qmodel                                        │
│                         │                                           │
│                         ▼                                           │
│                    Q-values                                        │
│                                                                     │
│              [Q(a0), Q(a1), ..., Q(a11)]                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       POLICY MODULE                                │
│                                                                     │
│                       Q-values                                     │
│                           │                                         │
│                  ┌────────┴────────┐                                │
│                  │                 │                                │
│              greedy            epsilon                             │
│                  │                 │                                │
│                  └────────┬────────┘                                │
│                           ▼                                         │
│                         action                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                         ENVIRONMENT
                               │
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
                  reward               frame
                     │                   │
                     │                   ▼
                     │             Observation
                     │                Module
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       REPLAY BUFFER                                 │
│                                                                     │
│       state1 ───────────────┐                                       │
│       action ───────────────┤                                       │
│       reward ───────────────┼──→ replay.add_memory()               │
│       state2 ───────────────┘                                       │
│                                                                     │
│       Later:                                                       │
│                                                                     │
│       replay.get_batch()                                           │
│               │                                                     │
│               ▼                                                     │
│       minibatch_train()                                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LOSS MODULE                                 │
│                                                                     │
│       Q loss                                                        │
│       Forward loss                                                  │
│       Inverse loss                                                  │
│             │                                                       │
│             ▼                                                       │
│       loss_fn()                                                     │
│             │                                                       │
│             ▼                                                       │
│       total loss                                                    │
│             │                                                       │
│             ▼                                                       │
│       backward()                                                    │
│             │                                                       │
│             ▼                                                       │
│       optimizer.step()                                              │
└─────────────────────────────────────────────────────────────────────┘
```

## The most important idea: separate the modules

I would organize your code roughly like this:

```text
mario_rl/
│
├── config.py
│
├── environment.py
│
├── preprocessing.py
│
├── state.py
│
├── policy.py
│
├── model.py
│
├── replay.py
│
├── trainer.py
│
└── train.py
```

### `preprocessing.py`

Only worry about:

```python
def downscale_obs(...):
    ...

def grayscale(...):
    ...

def prepare_state(...):
    ...
```

Then you can change:

```text
skimage
   ↓
OpenCV
   ↓
PyTorch
```

without touching the RL algorithm.

---

### `state.py`

Only worry about temporal state:

```python
class StateBuilder:

    def __init__(self, frames_per_state):
        ...

    def reset(self):
        ...

    def add_frame(self, frame):
        ...

    def get_state(self):
        ...
```

Then your training code doesn't care how frames are stored.

For example:

```text
frame
  ↓
StateBuilder.add_frame()
  ↓
deque
  ↓
stack
  ↓
state
```

---

### `policy.py`

Only action selection:

```python
def select_action(q_values, epsilon=0):
    ...
```

You can later experiment with:

```text
greedy
epsilon-greedy
softmax
Boltzmann
random
```

without touching the environment.

---

### `environment.py`

Put the repeated environment interaction here:

```python
def step_action(env, action, action_repeats):
    ...
```

Instead of your training loop containing:

```python
for j in range(params['action_repeats']):
    state2, e_reward_, done, trunc, info = env.step(action)
    ...
```

you could have:

```python
result = step_action(
    env,
    action,
    params['action_repeats']
)
```

and get:

```text
result
 ├── frame
 ├── reward
 ├── done
 ├── trunc
 └── info
```

Now you can modify action repetition independently.

---

# Most importantly: isolate the "anti-cheating" logic

I would give this its own module:

```text
progress_monitor.py
```

Something like:

```python
class ProgressMonitor:

    def __init__(
        self,
        min_progress,
        max_no_progress_steps
    ):
        self.min_progress = min_progress
        self.max_no_progress_steps = max_no_progress_steps

        self.best_x = 0
        self.no_progress_steps = 0

    def reset(self, x_pos=0):
        self.best_x = x_pos
        self.no_progress_steps = 0

    def update(self, x_pos):

        if x_pos >= self.best_x + self.min_progress:
            self.best_x = x_pos
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        return (
            self.no_progress_steps
            > self.max_no_progress_steps
        )
```

Then your main training loop simply says:

```python
stuck = progress_monitor.update(info['x_pos'])

if stuck:
    done = True
```

This is **much better architecturally**.

Now you can independently experiment with:

```text
ProgressMonitor A
    ↓
x-position based

ProgressMonitor B
    ↓
velocity based

ProgressMonitor C
    ↓
x-position + death + time

ProgressMonitor D
    ↓
reward-based
```

without modifying the training algorithm.

---

# Your final training loop becomes much cleaner

Instead of having 100+ lines mixed together:

```python
for i in range(epochs):

    # environment
    # rendering
    # resizing
    # state construction
    # action selection
    # progress checking
    # replay
    # minibatch
    # loss
    # optimization
    # logging
```

you get:

```python
for i in range(epochs):

    # 1. State → Q values
    q_values = Qmodel(state)

    # 2. Q values → action
    action = policy(q_values, epsilon)

    # 3. Action → environment result
    result = environment.step(action)

    # 4. Raw frame → processed frame
    frame = preprocessor(result.frame)

    # 5. Frame → next state
    next_state = state_builder.add(frame)

    # 6. Check whether agent is stuck
    done = progress_monitor.update(result.x_pos)

    # 7. Store experience
    replay.add_memory(
        state,
        action,
        result.reward,
        next_state
    )

    # 8. Train network
    train_step()

    # 9. Reset if necessary
    if done:
        reset_episode()
```

That's the structure I recommend you aim for.

The key principle is:

```text
                    TRAINING LOOP
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
  Preprocessor       Policy          Environment
       │                 │                 │
       ▼                 ▼                 ▼
    State            Action          Observation
       │                                   │
       └─────────────────┬─────────────────┘
                         ▼
                    Replay Buffer
                         │
                         ▼
                    Train Network
```

Then **each box can be replaced independently**.

For your current problem, I'd make `preprocessing.py` the first module to isolate, because then you can benchmark:

```text
skimage.resize
       VS
cv2.resize
       VS
torch.interpolate
```

without changing anything in the actual RL training logic.
