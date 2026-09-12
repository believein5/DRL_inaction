import multiprocessing as mp
import numpy as np


# ============================================================
# 版本 1：使用 mp.Pool + pool.map
# 优点：API 简洁，自动收集返回值（顺序与输入一致），自动管理进程生命周期
# 缺点：控制粒度低
# ============================================================
def square(x):
    return np.square(x)


# ============================================================
# 版本 2：手动创建 mp.Process + mp.Queue
# 优点：完全控制每个进程；可以通过 Queue 在进程间传数据
# 缺点：需要自己 start/join/terminate，结果顺序不保证（Queue 是先到先出）
# ============================================================
def square_with_queue(i, x, queue):
    print("In process {}".format(i))
    queue.put(np.square(x))


x = np.arange(64)


if __name__ == '__main__':
    print("CPU count:", mp.cpu_count())

    # ---------- 版本 1：Pool.map ----------
    print("\n--- Pool.map version ---")
    pool = mp.Pool(16)
    squared = pool.map(square, [x[8*i:8*i+8] for i in range(8)])
    pool.close()
    pool.join()
    print(squared)

    # ---------- 版本 2：手动 Process + Queue ----------
    print("\n--- Manual Process + Queue version ---")
    processes = []
    queue = mp.Queue()

    # 启动 8 个进程，每个处理一段数据
    for i in range(8):
        start_index = 8 * i
        proc = mp.Process(
            target=square_with_queue,
            args=(i, x[start_index:start_index+8], queue),
        )
        proc.start()
        processes.append(proc)

    # 等待所有子进程结束
    for proc in processes:
        proc.join()

    # 清理（join 之后其实进程已经结束了，terminate 是保险）
    for proc in processes:
        proc.terminate()

    # 从 queue 里把结果全部取出来
    results = []
    while not queue.empty():
        results.append(queue.get())
    print(results)
