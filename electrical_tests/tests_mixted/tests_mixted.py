from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
from hatt import HATT

dataset = "elec"

stream = FileStream("../"+dataset+".csv", n_targets=1, target_idx=-1)
# 2. Prepare for use
stream.prepare_for_use()

h = [
    HoeffdingTree(),
        HATT(),
]
for j in range(0,10000,1000):
    for i in range(1,100,10):
        evaluator = EvaluatePrequential(
            pretrain_size=0,
            show_plot=False,
            max_samples=20000,
            metrics=['accuracy','kappa'],
            batch_size=i,
            output_file="results/output_pretrain_"+str(j) +"_batch_size_" + str(i) + ".csv"
        )

        evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])
