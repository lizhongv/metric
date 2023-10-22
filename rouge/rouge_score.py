# pip install rouge
# https://github.com/pltrdy/rouge

# https://huggingface.co/spaces/evaluate-metric/rouge
# https://github.com/google-research/google-research/tree/master/rouge
# https://torchmetrics.readthedocs.io/en/stable/text/rouge_score.html




from rouge import Rouge
import json


def single_rouge_score():
    candidate = ['i am a student from xx school']  # 预测摘要, 可以是列表也可以是句子
    reference = ['i am a student from school on china'] #真实摘要
    # [
    #     {
    #         'rouge-1': {'r': 0.75, 'p': 0.8571428571428571, 'f': 0.7999999950222222}, 
    #         'rouge-2': {'r': 0.5714285714285714, 'p': 0.6666666666666666, 'f': 0.6153846104142012}, 
    #         'rouge-l': {'r': 0.75, 'p': 0.8571428571428571, 'f': 0.7999999950222222}
    #     }
    # ]

    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=candidate, refs=reference)
    print(rouge_score)  # json

    # print(rouge_score[0]["rouge-1"])
    # print(rouge_score[0]["rouge-2"])
    # print(rouge_score[0]["rouge-l"])  


def multi_rouge_score():
    
    # Load some sentences
    with open('./data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    hyps, refs = map(list, zip(*[[d['hyp'], d['ref']] for d in data]))

    a = [[h] for h in hyps]
    b = [[r] for r in refs]
    print(f"hyps size is {len(a)}, hpys is {a}\n")
    print(f"refs size is {len(b)}, refs is {b}\n")

    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs)
    print(scores)

    # [
    #     {
    #         'rouge-1': {'r': 0.4583333333333333, 'p': 0.6285714285714286, 'f': 0.5301204770503702}, 
    #         'rouge-2': {'r': 0.21739130434782608, 'p': 0.375, 'f': 0.2752293531520916}, 
    #         'rouge-l': {'r': 0.4166666666666667, 'p': 0.5714285714285714, 'f': 0.4819277059660328}
    #     }, 
    #     {
    #         'rouge-1': {'r': 0.20512820512820512, 'p': 0.32, 'f': 0.24999999523925787}, 
    #         'rouge-2': {'r': 0.019230769230769232, 'p': 0.030303030303030304, 'f': 0.023529407014533828}, 
    #         'rouge-l': {'r': 0.15384615384615385, 'p': 0.24, 'f': 0.18749999523925795}
    #     }, 
    #     {
    #         'rouge-1': {'r': 0.15789473684210525, 'p': 0.375, 'f': 0.22222221805212622}, 
    #         'rouge-2': {'r': 0.043478260869565216, 'p': 0.09090909090909091, 'f': 0.0588235250346024}, 
    #         'rouge-l': {'r': 0.07894736842105263, 'p': 0.1875, 'f': 0.11111110694101523}
    #     }, 
    #     {
    #         'rouge-1': {'r': 0.20588235294117646, 'p': 0.25925925925925924, 'f': 0.2295081917871541}, 
    #         'rouge-2': {'r': 0.027777777777777776, 'p': 0.030303030303030304, 'f': 0.028985502255829465}, 
    #         'rouge-l': {'r': 0.20588235294117646, 'p': 0.25925925925925924, 'f': 0.2295081917871541}
    #     }
    # ]

    # or
    scores = rouge.get_scores(hyps, refs, avg=True)
    print(scores)
    # {
    #     'rouge-1': {'r': 0.256809657061205, 'p': 0.395707671957672, 'f': 0.30796272053222706}, 
    #     'rouge-2': {'r': 0.07696952805648458, 'p': 0.13162878787878787, 'f': 0.09664194686426433}, 
    #     'rouge-l': {'r': 0.2138356354687624, 'p': 0.31454695767195767, 'f': 0.252511749983365}
    # }


if __name__ == "__main__":
    # single_rouge_score()
    multi_rouge_score()
   

   


