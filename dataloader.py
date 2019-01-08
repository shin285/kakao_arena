import h5py
import json
import random


# 각 분류 별 classifier 구현

# 카테고리 별로 8:2 비율로 나눔
# 각 데이터를 읽으면서 20% 확률로 validation data로 assign

def convert_utf8(x):
    return x.decode('UTF-8').strip()


def split_data(path):
    filename_list = [path + "/train.chunk.0" + str(i) for i in range(1, 10)]
    data_list = []
    for filename in filename_list:
        print(filename)
        data = h5py.File(filename, 'r')
        # bcateid = data['train/bcateid'].value[:22]
        # mcateid = data['train/mcateid'].value[:22]
        # scateid = data['train/scateid'].value[:22]
        # dcateid = data['train/dcateid'].value[:22]
        #
        # model = data['train/model'].value[:22]
        # brand = data['train/brand'].value[:22]
        # maker = data['train/maker'].value[:22]

        bcateid = data['train/bcateid'].value[:]
        mcateid = data['train/mcateid'].value[:]
        scateid = data['train/scateid'].value[:]
        dcateid = data['train/dcateid'].value[:]

        model = data['train/model'].value[:]
        brand = data['train/brand'].value[:]
        maker = data['train/maker'].value[:]

        data_list.extend(list(zip(bcateid, mcateid, scateid, dcateid, model, brand, maker)))

    total_len = len(data_list)
    validation_idx = total_len // 5

    return data_list[validation_idx:], data_list[:validation_idx]


def read_h5py(filename):
    data = h5py.File(filename, 'r')
    print(data['train/bcateid'].value[:])
    print(data['train/mcateid'].value[:])
    data.close()


# validation generator

# training generator


def yield_test(target_list):
    i = 0
    while True:
        yield target_list[i]
        i += 1
        if not i < len(target_list):
            break


def load_category_info(path):
    with open(path) as f:
        data = json.load(f)
    return data


def data_generator(data, batch_size=256):
    data_len = len(data)
    from_idx = 0
    to_idx = min(data_len, from_idx + batch_size)
    while True:
        values = data.value[from_idx:to_idx]

        # todo yield values

        if data_len == to_idx:
            break
        from_idx = from_idx + batch_size
        to_idx = from_idx + min(data_len - from_idx, batch_size)


def load_to_list(filenames):
    for filename in filenames:
        print(filename)
        data = h5py.File(filename, 'r')
        brand_data = data['train/bcateid']
        print(brand_data.value[:])
        train = data['train']
        bcate_id_dic = data_generator(train['bcateid'])

        brand_data = train['brand']
        model_data = train['model']
        maker_data = train['maker']

        bcate_ids = train['bcateid']
        mcate_ids = train['mcateid']
        scate_ids = train['scateid']
        dcate_ids = train['dcateid']

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


# get_data('D:\data\kakao_arena')


def multiple_yield_test():
    first_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    second_list = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    first_generator = yield_test(first_list)
    second_generator = yield_test(first_list)
    while True:
        yield next(first_generator), next(second_generator)


# g = multiple_yield_test()
#
# while True:
#     print(next(g))



# # split data to training data and validation data
# training_data, validation_data = split_data('D:\data\kakao_arena')
#
#
#
# print(len(training_data))
# print(len(validation_data))
