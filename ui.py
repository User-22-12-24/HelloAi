import threading
import tkinter
import random
from tkinter import messagebox

import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
# 实现默认的Matplotlib键绑定。
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt

import numpy as np

import trainer

train = trainer.Model('find_online', 2, 2, 1, results=('blue', 'red'))

matplotlib.rcParams['font.family'] = 'KaiTi'
matplotlib.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


root = tkinter.Tk()
root.wm_title("红蓝点分布")
root.geometry('+0+0')

fig = plt.Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
ax = fig.add_subplot()
ax.set_xlabel("时间")
ax.set_ylabel("损失率")

canvas = FigureCanvasTkAgg(fig, master=root)  # 在tk的绘图区
canvas.draw()

# 工具栏，pack_toolbar=False将使以后使用布局管理器更容易
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tkinter.Button(master=root, text="退出", command=exit)
data_x_try = tkinter.Entry(root)
data_y_try = tkinter.Entry(root)
run_try = tkinter.Button(
    root,
    text='测试',
    command=lambda: messagebox.showinfo(
        '结果',
        f"""期望结果为:{"blue" if int(data_y_try.get()) < -int(data_x_try.get())/2+3 else "red"}
结果是:{train["b"][0].get_best(int(data_x_try.get()), int(data_y_try.get()))}"""
    )
)


def get_data():
    data = set()
    x = -10
    while x < 10:
        x += random.uniform(0, 1)
        y = random.uniform(-10, 10)
        data.add(((x, y), 'blue' if y < -x/2+3 else 'red'))
    return data


def lost(func, data, i):
    s = 0
    data = list(data)
    ax.plot([0, 0], [0, 1], c='black', marker='.')
    for d in data:
        r = func(putin=d[0], return_result=False)
        dat[0].append(dat[0][-1]+1)
        if d[1] == 'blue':
            ls = (abs(1-r['blue'])+abs(r['red']))/2
        if d[1] == 'red':
            ls = (abs(1-r['red'])+abs(r['blue']))/2
        s += ls
        dat[1].append(ls)
        dat[2].append(train['b'][2])
        if len(dat[0]) > 2:
            dat[0].pop(0)
            dat[1].pop(0)
            dat[2].pop(0)
        ax.plot(dat[0], dat[1], c='red', label='本次损失率')
        ax.plot(dat[0], dat[2], c='blue', label='最低损失率')
        # 更新画布
        canvas.draw()
        if data.index(d) >= len(data)/10:
            train[i] = s/len(data)
    return s/len(data)


dat = [[0.0], [train['b'][2]], [train['b'][2]]]
ax.plot(dat[0], dat[1], c='red', label='本次损失率')
ax.plot(dat[0], dat[2], c='blue', label='最低损失率')
ax.legend()

datas = get_data()


def update_frequency():
    global dat, ax
    train.new_networks(2)
    for i in range(2):
        lost(train[i][0], datas, i)
        # 更新画布
        canvas.draw()


def update():
    update_frequency()
    while True:
        update_frequency()
        # time.sleep(1)
        pass


threading.Thread(target=update).start()

# 包装顺序很重要。
# 部件按顺序处理，如果由于窗口太小而没有剩余空间，则不显示它们。
# 画布的大小是相当灵活的，所以我们最后打包它，以确保UI控件显示尽可能长的时间。
canvas.get_tk_widget().pack(side=tkinter.TOP)
toolbar.pack()
tkinter.Label(root, text='x:').pack()
data_x_try.pack()
tkinter.Label(root, text='y:').pack()
data_y_try.pack()
run_try.pack()
button_quit.pack()

tkinter.mainloop()
