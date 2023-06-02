import os
import pickle
import random
import time

import neuralNetwork as Network


class Model:
    def __init__(self, model_name, input_layer, output_layer, *hidden_layer, **kwargs):
        self.model_name = model_name
        dirs = list(os.walk('.'))[0][1]
        print('[dirs]', dirs)
        if model_name not in dirs:
            os.makedirs(model_name)
            os.makedirs(model_name+'/newModel')
            with open(f'{self.model_name}/bestNow.model', 'wb') as f:
                net_work = Network.Network(input_layer, output_layer, hidden_layer=hidden_layer)
                if 'results' in kwargs:
                    net_work.results = kwargs['results']
                print('[output now]', net_work.results)
                pickle.dump([net_work, 0, 100000000], f)

    def __call__(self, *putin, **kwargs):
        if len(putin) == 0:
            putin = kwargs['putin']
        with open(f'{self.model_name}/bestNow.model', 'rb') as f:
            best = pickle.load(f)
        return best[0].get_best(putin=putin) if kwargs['return_result'] else best[0](putin=putin)

    def new_networks(self, number=100):
        with open(f'{self.model_name}/bestNow.model', 'rb')as f:
            best_now = pickle.load(f)
        print('='*20)
        print('='*20)
        print('[best now]', best_now)
        print('='*20)
        best_now[1] += 1
        for n in range(number):
            now = best_now
            for i in range(len(best_now[0].size)):
                for j in range(best_now[0].size[i]):
                    print('[weights last]', now[0][i, j, 0])
                    print('[threshold last]', now[0][i, j, 1])
                    print('[data last]', now)
                    now[0][i, j, 0] += random.uniform(
                        -number*10, number*10
                    )
                    now[0][i, j, 1] += random.uniform(
                        -number*10, number*10
                    )
                    print('[weights now]', now[0][i, j, 0])
                    print('[threshold now]', now[0][i, j, 1])
                    print('[data now]', now)
                    print()
            with open(f'{self.model_name}/newModel/{n}.model', 'wb') as f:
                pickle.dump(now, f)

    @property
    def networks(self):
        return list(os.walk(f'{self.model_name}/newModel'))[0][2]

    def __getitem__(self, item):
        try:
            with open(f'{self.model_name}/newModel/{item}.model', 'rb') as f:
                return pickle.load(f)
        except Exception:
            with open(f'{self.model_name}/bestNow.model', 'rb') as f:
                return pickle.load(f)

    def __setitem__(self, key, value):
        with open(f'{self.model_name}/bestNow.model', 'rb') as f:
            lose = pickle.load(f)[2]
        with open(f'{self.model_name}/newModel/{key}.model', 'rb') as f:
            this = pickle.load(f)
        this[2] = value
        if this[2] < lose:
            with open(f'{self.model_name}/bestNow.model', 'wb') as f:
                pickle.dump(this, f)
            length = len(list(os.walk(f'{self.model_name}/newModel/'))[-1])
            self.new_networks(length)
            time.sleep(1)


if __name__ == '__main__':
    m = Model('模型1', 2, 2, 3, 3, results=[print, input])
    m.new_networks(2)
    print(m[0][0](1, 2))
    m[0] = m[0][2]-10
    print(m(1, 2, return_result=True))
