# https://github.com/neulab/BARTScore

from bart_score import BARTScorer
bart_scorer = BARTScorer(device="cuda:1", checkpoint='../../bart-large')


def single_reference():

    output = bart_scorer.score(
        ['This is interesting.', 'Cats are great pets.', 'The sun rises in the east.', 'Reading books is a great way to gain knowledge.'],
        ['This is interestion.', 'Dogs are great pets.', 'The east is where the sun comes up.', 'The act of reading serves no purpose, ngaging in other activities would be more meaningful.'],
        batch_size=2,
    )
    print(output) # [-6.643669128417969, -3.5261921882629395, -5.329223155975342, -8.365194320678711]
    # log p 与 p同变换，所以值越大得分越高，即绝对值越小越相似


def multi_reference():

    srcs = ["I'm super happy today.", "This is a good idea.", "Reading makes people smarter."]
    tgts = [
        ["I feel good today.", "I feel sad today.", "I'm absolutely thrilled today."],  # src中第一个与该组所有进行比较
        ["Not bad.", "Sounds like a good idea.",  "I'm in favor of this idea."],  # src中第二个与该组所有进行比较
        ["Time flies by so quickly.", "The sun rises in the east.", "Reading is utterly useless."] # 语义完全不同反例，src中第三个句子与改组的所有进行比较
    ] 
    # List[List of references for each test sample]

    output = bart_scorer.multi_ref_score(
        srcs,
        tgts,
        agg='max', # mean, max
        index=True,
        batch_size=2,
    )
    print(output)


if __name__ == "__main__":
    # single_reference()
    multi_reference()