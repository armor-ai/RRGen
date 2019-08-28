# Visualize attention layer

import os, pickle
import matplotlib.pyplot as plt
import json
import argparse
import numpy as np

att_fp = 'C:\\Users\\student\\Downloads\\attention\\0.34927394454_6'
att_fr = open(att_fp, 'rb')
lines = pickle.load(att_fr, encoding='latin1')
att_fr.close()

text_fp = 'C:\\Users\\student\\Downloads\\texts\\0.34927394454_6'
text_fr = open(text_fp)
texts = text_fr.readlines()
text_fr.close()


def plot_heat_maps(mma, target_labels, source_labels, id):
	fig, ax = plt.subplots()
	heatmap = ax.pcolor(mma, cmap=plt.cm.Blues, linewidths=4)

	# put the major ticks at the middle of each cell
	ax.set_xticks(np.arange(mma.shape[1]) + 0.5, minor=False)
	ax.set_yticks(np.arange(mma.shape[0]) + 0.5, minor=False)

	# without this I get some extra columns rows
	# http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
	ax.set_xlim(0, int(mma.shape[1]))
	ax.set_ylim(0, int(mma.shape[0]))

	# want a more natural, table-like display
	ax.invert_yaxis()
	ax.xaxis.tick_top()

	# source words -> column labels
	ax.set_xticklabels(source_labels, minor=False)
	# target words -> row labels
	ax.set_yticklabels(target_labels, minor=False)

	plt.xticks(rotation=45, fontsize=14)
	plt.yticks(fontsize=14)
	fig.colorbar(heatmap, ax=ax)
	# plt.tight_layout()
	plt.show()
	# plt.savefig('../result/'+str(id)+'.pdf')

## Output (out_text_len, input_text_len), i.e., (reply_len, review_len+1)
att_vectors = []
for idx, line in enumerate(lines):
	att_vector = line
	att_vectors.append(att_vector)
	text = texts[idx].split('**')
	review = text[0]+' </s>'
	reply = text[1]
	print(att_vector.shape, len(reply.split()), len(review.split()))
	break

## 12, 190,194 for BLEU at 0.315427058064
# oid = 86
# plot_heat_maps(np.delete(lines[oid], np.s_[-1:], axis=1).transpose(), (texts[oid].split('**')[0]).split(), (texts[oid].split('**')[1]).split(), oid)
xid = 87
mat = np.delete(np.delete(lines[xid], np.s_[-1:], axis=1).transpose(), np.s_[-12:], axis=1)
mat = np.delete(mat, np.s_[:7], axis=1)
plot_heat_maps(mat, (texts[xid].split('**')[0]).split(), (texts[xid].split('**')[1]).split()[7:-12], xid)
# yid = 88
# plot_heat_maps(np.delete(lines[yid], np.s_[-1:], axis=1).transpose(), (texts[yid].split('**')[0]).split(), (texts[yid].split('**')[1]).split(), yid)