from operator import itemgetter

import numpy as np
from collections import Counter, OrderedDict
import h5py

bid_counts = dict()


def get_ids(id_data):
    id_data_len = len(id_data)
    print("total len : ", id_data_len)
    from_idx = 0
    to_idx = min(id_data_len, from_idx + 100000)
    while True:
        id_values = id_data.value[from_idx:to_idx]
        for i in id_values:
            bid_counts[i] = bid_counts.get(i, 0) + 1
        print(from_idx, ":", to_idx)
        print(id_values)
        if id_data_len == to_idx:
            break
        from_idx = from_idx + 100000
        to_idx = from_idx + min(id_data_len - from_idx, 100000)


def load_to_list(filenames):
    for filename in filenames:
        print(filename)
        data = h5py.File(filename, 'r')
        train = data['train']

        bcate_id_dic = get_ids(train['bcateid'])
        print()

        # brand_data = train['brand']
        # maker_data = train['maker']
        bcate_ids = train['bcateid']
        # print(bcate_ids.value[0])
        # mcate_ids = train['mcateid']
        # scate_ids = train['scateid']
        # dcate_ids = train['dcateid']

        # train_keys = train.keys()
        #
        # for key in train_keys:
        #     if key == 'img_feat':
        #         continue
        #     train_data = train[key]
        #     print(type(train_data))
        #
        #     print("total length : ", str(len(train_data.value)))
        #     value = train_data.value[0:100000]
        #     if isinstance(value[0], np.int32):
        #         pass
        #     else:
        #         value_string = [x.decode('UTF-8') for x in value]
        #         pass
        # print()


def get_data(path):
    filenames = [path + "/train.chunk.0" + str(i) for i in range(1, 10)]
    total_data_list = load_to_list(filenames)


get_data("D:/data/kakao_arena")

import matplotlib.pyplot as plt

print(bid_counts)
aa = {k: v for k, v in sorted(bid_counts.items(), key=lambda x: x[1])}

plt.bar(list(aa.keys()), aa.values(), color='b')
plt.show()
