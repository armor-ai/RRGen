import numpy as np
import os
import json
from collections import defaultdict
from parameter import opts
from ast import literal_eval
from nltk.stem.wordnet import WordNetLemmatizer


class LoadExtFeature():
    def __init__(self):
        feature_dir = opts.external_dataset
        self.senti_fn = os.path.join(feature_dir, 'reviews_sentiment.txt')
        self.cate_fn = os.path.join(feature_dir, 'app_category.txt')
        self.rating_fn = os.path.join(feature_dir, 'single_test_data.txt')
        self.app_cate_dict = self._get_cate()

    def _get_senti(self):
        senti_pos_dict = {}
        senti_neg_dict = {}
        senti_fr = open(self.senti_fn)
        lines = senti_fr.readlines()
        senti_fr.close()
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            terms = line.split('\t')
            senti_pos_dict[idx-1] = int(terms[0])
            senti_neg_dict[idx-1] = int(terms[1])+5
        return senti_pos_dict, senti_neg_dict

    def _get_cate(self):
        # get app category
        cate_fr = open(self.cate_fn)
        app_cate_dict = literal_eval(cate_fr.readlines()[0])
        cate_fr.close()
        cates = list(set(app_cate_dict.values()))
        for app, cate in app_cate_dict.items():
            app_cate_dict[app] = cates.index(cate)
        return app_cate_dict

    def _get_review_cate(self):
        # get review category
        pass

    def _get_rating(self):
        rating_dict = {}
        cate_dict = {}
        rating_fr = open(self.rating_fn)
        app_infos = rating_fr.readlines()
        rating_fr.close()
        for idx, app_info in enumerate(app_infos):
            terms = app_info.split('***')
            rating_dict[idx] = int(terms[1])
            cate_dict[idx] = int(self.app_cate_dict[terms[0].strip()])
        return rating_dict, cate_dict

def _get_keywords(fr):
    keyword_dict = {}
    lines = fr.readlines()
    fr.close()
    for line in lines:
        cate, keywords = line.split("\t")
        keywords = keywords.split(",")
        for idx, key in enumerate(keywords):
            keywords[idx] = WordNetLemmatizer().lemmatize(key.strip("\r\n"), "v")
            keywords[idx] = WordNetLemmatizer().lemmatize(keywords[idx], "n")
        keyword_dict[cate] = keywords
    fw = open("/research/lyu1/cygao/workspace/data/keyword_dict.json", "w")
    json.dump(keyword_dict, fw)
    fw.close()

if __name__ == "__main__":
    # app_cate_dict = LoadExtFeature()._get_cate()
    # senti_pos_dict, senti_neg_dict = LoadExtFeature()._get_senti()
    # rating_dict, cate_dict = LoadExtFeature()._get_rating()
    # print(set(cate_dict.values()), len(set(cate_dict.values())))
    #
    # cate_arr = np.eye(np.max(cate_dict.values()))[np.subtract(cate_dict.values(), 1)]
    # rating_arr = np.eye(5)[np.subtract(rating_dict.values(), 1)]
    # senti_pos_arr = np.eye(max(senti_pos_dict.values()))[np.subtract(senti_pos_dict.values(), 1)]
    # senti_neg_arr = np.eye(max(senti_neg_dict.values()))[np.subtract(senti_neg_dict.values(), 1)]
    # print(senti_pos_arr[:2, :], senti_neg_arr[:2, :])

    keyword_fr = open("/research/lyu1/cygao/workspace/data/cate_keyword.txt")
    _get_keywords(keyword_fr)

