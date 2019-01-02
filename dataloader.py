import numpy as np

import h5py


def load_to_list(filenames):
    for filename in filenames:
        print(filename)
        data = h5py.File(filename, 'r')
        train = data['train']
        train_keys = train.keys()
        for key in train_keys:
            if key == 'img_feat':
                continue
            train_data = train[key]

            print("total length : ", str(len(train_data.value)))
            value = train_data.value[0:100000]
            if isinstance(value[0], np.int32):
                pass
            else:
                value_string = [x.decode('UTF-8') for x in value]
                pass
        print()


def get_data(path):
    filenames = [path + "/train.chunk.0" + str(i) for i in range(1, 10)]
    total_data_list = load_to_list(filenames)


get_data("D:/data/kakao_arena")