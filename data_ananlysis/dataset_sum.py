# Analyze the datasets
from argparse import Namespace
from tqdm import tqdm
from ast import literal_eval
import numpy as np


args = Namespace(
# glove_filename='D:\Github\Reply_reviews\glove.6B\glove.6B.50d.txt',
#     test_filename='D:\Github\Reply_reviews\data\single_test_data.txt',
#     train_filename='D:\Github\Reply_reviews\data\single_test_data.txt',
#     valid_filename='D:\Github\Reply_reviews\data\single_test_data.txt'
    glove_filename='/home/cuiyun1/tasks/review_reply/data/glove.6B.100d.txt',
    test_filename='/home/cuiyun1/tasks/review_reply/data/single_test_data.txt',
    train_filename='/home/cuiyun1/tasks/review_reply/data/single_train_data.txt',
    valid_filename='/home/cuiyun1/tasks/review_reply/data/single_valid_data.txt',
    # test_filename='/research/lyu1/cygao/workspace/data/single_test_senti.txt',
    # train_filename='/research/lyu1/cygao/workspace/data/single_train_senti.txt',
    # valid_filename='/research/lyu1/cygao/workspace/data/single_valid_senti.txt',
    cate_filename='/home/cuiyun1/tasks/review_reply/data/app_category.txt',
    keyword_filename='/home/cuiyun1/tasks/review_reply/data/keyword_dict.json',
    feature_vec_size=20
)




def load_cates(catepath):
        cate_fr = open(catepath)
        app_cate_dict = literal_eval(cate_fr.readlines()[0])
        cate_fr.close()
        return app_cate_dict

def get_cate_num(fpath, cate_nums, app_nums, total_data):
    with open(fpath) as file:
        for sent in tqdm(file.readlines()):
            terms = sent.split('***')
            if len(terms) < 8:  # check term length
                continue
            app = terms[0]
            if app not in app_nums:
                app_nums[app] = 0
            app_nums[app] += 1
            cate = app_cate_dict[app]
            if cate not in cate_nums:
                cate_nums[cate] = 0
            cate_nums[cate] += 1
            total_data += 1
    return cate_nums, app_nums, total_data

def get_response_word_length(f_ls):
    # Check response lengths
    res_lens = []
    for fp in f_ls:
        with open(fp) as tf:
            for sent in tqdm(tf.readlines()):
                terms = sent.split('***')
                res_lens.append(len(terms[5].split()))
    return res_lens

if __name__ == "__main__":
    app_cate_dict = load_cates(args.cate_filename)
    cate_nums = {}
    app_nums = {}
    total_data = 0
    response_lens = get_response_word_length([args.train_filename, args.valid_filename, args.test_filename])
    print(len(response_lens), np.percentile(response_lens, 25), np.percentile(response_lens, 50))

    cate_nums, app_nums, train_num = get_cate_num(args.train_filename, cate_nums, app_nums, total_data)
    print("Num in training file is ", train_num)
    cate_nums, app_nums, valid_num = get_cate_num(args.valid_filename, cate_nums, app_nums, total_data)
    print("Num in valid file is ", valid_num)
    cate_nums, app_nums, test_num = get_cate_num(args.test_filename, cate_nums, app_nums, total_data)
    print("Num in test file is ", test_num)
    print("Totoal review num is ", train_num+valid_num+test_num)
    print("Num of apps is ", len(app_nums.keys()))
    print(cate_nums)
    print(app_nums)


