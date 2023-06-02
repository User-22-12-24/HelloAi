import random
import numpy as np


class Neural:
    WEIGHTS = 0  # 权重
    EIGEN = 1  # 特征

    def __init__(self, num):
        self._num = num
        self._function = {
            self.WEIGHTS: np.random.random(self._num) * random.uniform(1, 10),
            self.EIGEN: np.ones(self._num)
        }
        self._threshold = 0
        print('[function', self._function)

    def __call__(self, *eigen):
        if np.array(eigen).size == np.ones(self._num).size:
            self._function[self.EIGEN] = np.array(eigen)
            if self._function[self.WEIGHTS].size == self._function[self.EIGEN].size:
                print('[weights]', self._function[self.WEIGHTS])
                print('[threshold]', self._threshold)
                return (self._function[self.WEIGHTS] * self._function[self.EIGEN]).sum()+self._threshold
            else:
                raise ValueError(
                    'The size of the eigen must be {}, not {}!'.format(
                        self._function[self.WEIGHTS].size,
                        self._function[self.EIGEN].size
                    )
                )
        else:
            raise ValueError(
                f'The len of eigen in {self} must be {self._num}, not {len(eigen)}.'
            )

    @property
    def weights(self):
        return self._function[0]

    @weights.setter
    def weights(self, values):
        if len(values) == self._num:
            try:
                self._function[self.WEIGHTS] = np.array(values, dtype=float)
                print('[function now]', self._function)
            except ValueError:
                raise TypeError('All the values of the parameter "values" must be int or float!')
        else:
            raise ValueError(
                'The number of values of parameter "values" must be {}, the number of the weights.'.format(
                    len(self._function[self.WEIGHTS])
                )
            )

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """
        :param value: It's a parameter to set the threshold of this neural.
        """
        if isinstance(value, int) or isinstance(value, float):
            self._threshold = float(value)
        else:
            s = "'"
            raise TypeError(
                f'Parameter "value" should be int or float, not {str(type(value)).split(s)[1]}!'
            )


class Network:
    def __init__(self, input_layer, output_layer, *hidden_layer, **hidden):
        self._inl = np.zeros(input_layer)
        if len(hidden_layer) != 0:
            hidden_layer = np.array(hidden_layer)
        else:
            hidden_layer = hidden['hidden_layer']
        print('[hidden layer]', hidden_layer)
        self._oul = [
            [i for i in range(output_layer)],
            np.array([Neural(hidden_layer[-1]) for _ in range(output_layer)])
        ]
        self._hil = np.array(
            [
                np.array(
                    [
                        Neural(
                            hidden_layer[i-1]
                        ) if i != 0 else Neural(input_layer) for _ in range(hidden_layer[i])
                    ],
                    dtype=object
                ) for i in range(len(hidden_layer))
            ],
            dtype=object
        )
        self._layer_size = np.array(list(hidden_layer) + [output_layer])
        print('[layer size]', self._layer_size)
        print('[hidden layer]', self._hil)

    def __call__(self, *putin, **kwargs):
        if len(putin) == 0:
            putin = kwargs['putin']
        if self._inl.size == np.array(putin).size:
            self._inl = np.array(putin)
            net = list(self._hil)
            net.append(self._oul[1])
            print('[net]', net)
            for i in range(len(net)):
                net[i] = [neural(putin if i == 0 else net[i-1]) for neural in net[i]]
            output = {}
            for i in range(len(net[-1])):
                output[self._oul[0][i]] = net[-1][i]
            s = sum(list(output.values()))
            for k in output:
                output[k] = output[k]/s
            return output
        else:
            raise ValueError(
                f'The size of parameter "putin" must be {self._inl.size}, not {np.array(putin).size}'
            )

    def get_best(self, *putin, **kwargs):
        if len(putin) == 0:
            putin = kwargs['putin']
        result = list(self(putin).items())

        def a(x):
            return x[1]
        result.sort(reverse=True, key=a)
        print('[result]', result)
        return result[0][0]

    @property
    def size(self):
        return self._layer_size

    @property
    def results(self):
        return self._oul[0]

    @results.setter
    def results(self, results):
        if len(results) == len(self._oul[0]):
            self._oul[0] = list(results)
            print('[out layer]', self._oul)
        else:
            raise ValueError(
                f'The len of parameter "result" must be {len(self._oul[0])}, not {len(results)}'
            )

    def __getitem__(self, key):
        if key[0] == self._layer_size.size-1:
            if key[1] <= self._layer_size[-1]-1:
                if key[2] == 0:
                    return self._oul[1][key[1]].weights
                else:
                    return self._oul[1][key[1]].threshold
            else:
                raise ValueError(
                    f"""There is only {
                        self._layer_size[-1]
                    } in the hidden layer, the second item should at least 1 less than it(less than {
                        self._layer_size[-1]
                    })."""
                )
        elif key[0] < self._layer_size.size-1:
            if key[1] <= self._layer_size[key[0]]-1:
                if key[2] == 0:
                    return self._hil[key[:-1]].weights
                else:
                    return self._hil[key[:-1]].threshold
            else:
                raise ValueError(
                    f"""There is only {
                        self._layer_size[key[0]]
                    } in the hidden layer {
                        key[0]
                    }, the second item should at least 1 less than it(less than {
                        self._layer_size[key[0]]
                    })."""
                )
        else:
            raise ValueError(
                f"""There is only {
                    self._layer_size.size
                } layers, the first item should at least 1 less than it(less than {
                        self._layer_size.size
                })."""
            )

    def __setitem__(self, key, value):
        if key[0] == self._layer_size.size-1:
            if key[1] <= self._layer_size[-1]-1:
                if key[2] == 0:
                    self._oul[1][key[1]].weights = np.array(value, dtype=float)
                else:
                    self._oul[1][key[1]].threshold = value
            else:
                raise ValueError(
                    f"""There is only {
                        self._layer_size[-1]
                    } in the hidden layer, the second item should at least 1 less than it(less than {
                        self._layer_size[-1]
                    })."""
                )
        elif key[0] < self._layer_size.size-1:
            if key[1] <= self._layer_size[key[0]]-1:
                if key[2] == 0:
                    self._hil[key[:-1]].weights = np.array(value, dtype=float)
                else:
                    self._hil[key[:-1]].threshold = value
            else:
                raise ValueError(
                    f"""There is only {
                        self._layer_size[key[0]]
                    } in the hidden layer {
                        key[0]
                    }, the second item should at least 1 less than it(less than {
                        self._layer_size[key[0]]
                    })."""
                )
        else:
            raise ValueError(
                f"""There is only {
                    self._layer_size.size
                } layers, the first item should at least 1 less than it(less than {
                        self._layer_size.size
                })."""
            )


if __name__ == '__main__':
    n = Network(2, 2, 3, 3)
    n.results = (print, input)
    # n.get_best(2, 2)('hhh')
    print(n[1, 2, 0])
    n[1, 2, 0] = 2, 3, 5
    print(n[1, 2, 0])
