# https://github.com/Tiiiger/bert_score
# https://huggingface.co/spaces/evaluate-metric/bertscore
# 下载模型放在同级目录下

# pip install bert-score
# pip install evaluate

from bert_score import score
from evaluate import load


def single_ref():
    

    # data
    cands = [
        '我们都曾经年轻过，虽然我们都年少，但还是懂事的',
        '我们都曾经年轻过，虽然我们都年少，但还是懂事的'
        ]
    refs = [
        '虽然我们都年少，但还是懂事的',
        '我们都曾经年轻过，虽然我们都年少，但还是懂事的'
        ]


    P, R, F1 = score(cands, refs, lang="zh", verbose=True)
    result = score(cands, refs, lang="zh", verbose=True)
    print(f"P is {P}, R is {R}, F1 is {F1}") # P is tensor([0.8792, 1.0000]), R is tensor([0.9533, 1.0000]), F1 is tensor([0.9148, 1.0000])

    print(f"System level F1 score: {F1.mean():.3f}") 
    # System level F1 score: 0.957


# def use_huggingface():
#     bertscore = load("bertscore")

#     predictions = [
#          '我们都曾经年轻过，虽然我们都年少，但还是懂事的',
#         '我们都曾经年轻过，虽然我们都年少，但还是懂事的'
#     ]
#     references = [
#          '虽然我们都年少，但还是懂事的',
#         '我们都曾经年轻过，虽然我们都年少，但还是懂事的'
#     ]  
    
#     results = bertscore.compute(
#         predictions=predictions, 
#         references=references, 
#         lang="zh",# "en"
#         )
#     print(results)

if __name__ == "__main__":
    single_ref()
    # use_huggingface() # 太慢
