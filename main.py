import os
import random
import trainer


def get_data():
    data = set()
    x = -10
    while x < 10:
        x += random.uniform(0, 0.01)
        y = random.uniform(-10, 10)
        data.add(((x, y), 'blue' if y < -x/2+3 else 'red'))
    return data


def lost(func, data):
    s = 0
    for d in data:
        r = func(putin=d[0], return_result=False)
        if d[1] == 'blue':
            s += abs(r['blue']-1)-r['red']
        if d[1] == 'red':
            s += abs(r['red']-1)-r['blue']
    return s/len(data)


print(get_data())

train = trainer.Model('find_online', 2, 2, 2, results=('blue', 'red'))

for _ in range(1):
    train.new_networks(10)
    for i in range(10):
        l = lost(train[i][0], get_data())
        train[i] = l
        print('[lost]', l)

print(train(tuple(map(int, input().split())), return_result=True))

os.system('pause')
