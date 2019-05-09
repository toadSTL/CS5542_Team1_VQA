from nltk.translate.bleu_score import sentence_bleu
import json
import numpy
from statistics import mean

input_file = open ('validation_data.json')
answerData = json.load(input_file)
#len(answerData['data'])
num = 772

bleu_c = numpy.empty(num, dtype=float)
bleu_1 = numpy.empty(num, dtype=float)
bleu_2 = numpy.empty(num, dtype=float)
bleu_3 = numpy.empty(num, dtype=float)
bleu_4 = numpy.empty(num, dtype=float)

print(num)
itr = 0
for item in answerData['data']:
    if((item['actual_answer'] != "UNK") & (item['predicted_answer'] != "UNK")):
        ref = item['actual_answer']
        hyp = item['predicted_answer']
        bleu_c[itr] = sentence_bleu(ref, hyp)
        bleu_1[itr] = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0))
        bleu_2[itr] = sentence_bleu(ref, hyp, weights=(0, 1, 0, 0))
        bleu_3[itr] = sentence_bleu(ref, hyp, weights=(0, 0, 1, 0))
        bleu_4[itr] = sentence_bleu(ref, hyp, weights=(0, 0, 0, 1))
        itr += 1

print(mean(bleu_c))
print(mean(bleu_1))
print(mean(bleu_2))
print(mean(bleu_3))
print(mean(bleu_4))