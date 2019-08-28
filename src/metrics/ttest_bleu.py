## Compute the statistical test result between our model and NMT
from nmt_bleu import compute_bleu
import scipy.stats



rrgen_res_fp = '/research/lyu1/cygao/workspace/texts/0.350670918106_8'
nmt_res_fp = '/research/lyu1/cygao/workspace/texts_no/0.239541174571_7'
source_fp = '/research/lyu1/cygao/workspace/data/test_label.txt'

def read_gen_texts(fp):
    ## Return candidates
    candidates = []
    fr = open(fp)
    lines = fr.readlines()
    fr.close()
    for idx, line in enumerate(lines):
        candidates.append(lines[idx].split('**')[1].strip().split())
    return candidates

def read_sources(fp, rrgen_res_fp, nmt_res_fp):
    review_refs_dict = {}
    fr = open(fp)
    lines = fr.readlines()
    fr.close()
    for idx, line in enumerate(lines):
        terms = lines[idx].split('***')
        review_refs_dict[terms[4]] = [terms[5].strip().split()]

    rrgen_cands = []
    references = []
    rrgen_fr = open(rrgen_res_fp)
    rrgen_lines = rrgen_fr.readlines()
    rrgen_fr.close()
    for idx, line in enumerate(rrgen_lines):
        terms = rrgen_lines[idx].split('**')
        rrgen_cands.append(terms[1].strip().split())
        if terms[0] in review_refs_dict:
            references.append(review_refs_dict[terms[0]])

    nmt_cands = []
    nmt_fr = open(nmt_res_fp)
    nmt_lines = nmt_fr.readlines()
    nmt_fr.close()
    for idx, line in enumerate(nmt_lines):
        terms = nmt_lines[idx].split('**')
        nmt_cands.append(terms[1].strip().split())
    return references, rrgen_cands, nmt_cands

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

if __name__ == "__main__":
    n = 1550
    # references = list(chunks(read_sources(source_fp), n))
    # rrgen_cands = list(chunks(read_gen_texts(rrgen_res_fp), n))
    # nmt_cands = list(chunks(read_gen_texts(nmt_res_fp), n))
    references, rrgen_cands, nmt_cands = read_sources(source_fp, rrgen_res_fp, nmt_res_fp)
    references = list(chunks(references, n))
    rrgen_cands = list(chunks(rrgen_cands, n))
    nmt_cands = list(chunks(nmt_cands, n))

    rrgen_bleus = []
    nmt_bleus = []
    rrgen_p4 = []
    rrgen_p3 = []
    rrgen_p2 = []
    rrgen_p1 = []
    nmt_p4 = []
    nmt_p3 = []
    nmt_p2 = []
    nmt_p1 = []
    print(len(references), len(rrgen_cands), len(nmt_cands))
    for i in range(len(references)):
        rrgen_bleu, pls, _, _, _, _ = compute_bleu(references[i], rrgen_cands[i])
        rrgen_p4.append(pls[0])
        rrgen_p3.append(pls[1])
        rrgen_p2.append(pls[2])
        rrgen_p1.append(pls[3])
        rrgen_bleus.append(rrgen_bleu)

        nmt_bleu, nmt_pls, _, _, _, _ = compute_bleu(references[i], nmt_cands[i])
        nmt_p4.append(nmt_pls[0])
        nmt_p3.append(nmt_pls[1])
        nmt_p2.append(nmt_pls[2])
        nmt_p1.append(nmt_pls[3])
        nmt_bleus.append(nmt_bleu)

    print(rrgen_bleus, nmt_bleus)
    ttest_res = scipy.stats.ttest_ind(rrgen_bleus, nmt_bleus)
    mantest_res = scipy.stats.mannwhitneyu(rrgen_bleus, nmt_bleus)
    signtest_res = scipy.stats.wilcoxon(rrgen_bleus, nmt_bleus)
    print('Paired t-test result is ', ttest_res[1], scipy.stats.ttest_ind(rrgen_p4, nmt_p4)[1], scipy.stats.ttest_ind(rrgen_p3, nmt_p3)[1], scipy.stats.ttest_ind(rrgen_p2, nmt_p2)[1], scipy.stats.ttest_ind(rrgen_p1, nmt_p1)[1])
    print('Manwhitneyu test result is ', mantest_res[1], scipy.stats.mannwhitneyu(rrgen_p4, nmt_p4)[1], scipy.stats.mannwhitneyu(rrgen_p3, nmt_p3)[1], scipy.stats.mannwhitneyu(rrgen_p2, nmt_p2)[1], scipy.stats.mannwhitneyu(rrgen_p1, nmt_p1)[1])
    print('Signed-rank test result is ', signtest_res[1], scipy.stats.wilcoxon(rrgen_p4, nmt_p4)[1], scipy.stats.wilcoxon(rrgen_p3, nmt_p3)[1], scipy.stats.wilcoxon(rrgen_p2, nmt_p2)[1], scipy.stats.wilcoxon(rrgen_p1, nmt_p1)[1])