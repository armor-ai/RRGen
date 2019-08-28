### compute blue score

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import os


# ============================================
# NLTK Sentence bleu score
# ============================================

# blue_1 = []
# blue_2 = []
# blue_3 = []
# blue_4 = []
# for idx in range(len(references)):
#     blue_1.append(sentence_bleu(references[idx], candidates[idx], weights=(1, 0, 0, 0)))
#     blue_2.append(sentence_bleu(references[idx], candidates[idx], weights=(0.5, 0.5, 0, 0)))
#     blue_3.append(sentence_bleu(references[idx], candidates[idx], weights=(0.33, 0.33, 0.33, 0)))
#     blue_4.append(sentence_bleu(references[idx], candidates[idx], weights=(0.25, 0.25, 0.25, 0.25)))
#
# print('Cumulative 1-gram: %f' % (sum(blue_1)*100/float(len(blue_1))))
# print('Cumulative 2-gram: %f' % (sum(blue_2)*100/float(len(blue_2))))
# print('Cumulative 3-gram: %f' % (sum(blue_3)*100/float(len(blue_3))))
# print('Cumulative 4-gram: %f' % (sum(blue_4)*100/float(len(blue_4))))



# ============================================
# NLTK Doc bleu score
# ============================================

# bleu = bleu_score.corpus_bleu(references, candidates, smoothing_function=SmoothingFunction().method1)
# print('Doc BLEU score', bleu)



# ============================================
# Another sentence bleu score
# ============================================
import sys
sys.path.append("./metrics/")
from sentence_bleu import SentenceBleuScorer
# scorer = SentenceBleuScorer('n=4')
# print(references[0][0])
# print(candidates[0])
# scorer.set_reference(references[0][0])
# print(scorer.score(candidates[0]))

# ============================================
# NMT bleu score
# ============================================
from nmt_bleu import compute_bleu
# print("NMT BLEU score result is ", compute_bleu(references, candidates))



out_dir = '/research/lyu1/cygao/workspace/data/pred'
pred_dirs = os.listdir(out_dir)
all_pred_fw = open(os.path.join(out_dir, 'overall'), 'w')
# bleu_res = []
for pred_dir in pred_dirs:
    pred_fns = os.listdir(os.path.join(out_dir, pred_dir))
    for pred_fn in pred_fns:
        pred_path = os.path.join(out_dir, pred_dir, pred_fn)
        result_fr = open(pred_path)
        print("Current prediction path is ", pred_path)
        results = result_fr.readlines()
        result_fr.close()

        references = []
        candidates = []
        for result in results:
            terms = result.split("***")
            references.append([terms[1].split()])
            candidates.append(terms[2].split())

        bleu_4, pls, _, _, _, _ = compute_bleu(references, candidates)
        # bleu_res.append(pred_dir+'\t'+pred_fn+'\t'+str(bleu_4)+'\t'+' '.join([str(i) for i in pls])+'\n')
        all_pred_fw.write(pred_dir+'\t'+pred_fn+'\t'+str(bleu_4)+'\t'+' '.join([str(i) for i in pls]))
all_pred_fw.close()
