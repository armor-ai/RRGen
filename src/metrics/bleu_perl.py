## Use multi_bleu.perl for evaluation
## perl multi-bleu.perl reference.txt < translation.txt

import os
import subprocess

def get_ref_cand(opt_fn, pred_fn):
    result_fr = open(os.path.join('/research/lyu1/cygao/workspace/data', 'pred', opt_fn, pred_fn))
    results = result_fr.readlines()
    result_fr.close()

    references = []
    candidates = []
    for result in results:
        terms = result.split("***")
        references.append(terms[1]+'\n')
        candidates.append(terms[2])

    ref_fn = os.path.join('/research/lyu1/cygao/workspace/data', 'pred', opt_fn, 'ref_'+pred_fn)
    ref_fw = open(ref_fn, 'w')
    ref_fw.writelines(references)
    ref_fw.close()

    cad_fn = os.path.join('/research/lyu1/cygao/workspace/data', 'pred', opt_fn, 'cad_'+pred_fn)
    cad_fw = open(cad_fn, 'w')
    cad_fw.writelines(candidates)
    cad_fw.close()

    # run command
    output = subprocess.check_output(['perl', './metrics/multi-bleu.perl', ref_fn, '<', cad_fn])
    print(output)

    os.remove(ref_fn)
    os.remove(cad_fn)

    # ref file: os.path.join('/research/lyu1/cygao/workspace/data', 'pred', opt_fn, 'ref_'+pred_fn)
    # cand file: os.path.join('/research/lyu1/cygao/workspace/data', 'pred', opt_fn, 'cad_'+pred_fn)