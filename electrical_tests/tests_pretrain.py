from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

from hatt import HATT

# 1. Create a stream
dataset = "elec"

stream = FileStream("./"+dataset+".csv", n_targets=1, target_idx=-1)
# 2. Prepare for use
stream.prepare_for_use()


h = [
    HoeffdingTree(),
	HATT(),
]

for i in range(0,10000,100):
	evaluator = EvaluatePrequential(
	    pretrain_size=i,
	    show_plot=True,
	    max_samples=20000,
	    metrics=['accuracy','kappa'],
	    batch_size=1,
      output_file="output_pretrain_"+ str(i)+"_batch_size_" + "1" + "_dataset_"+dataset + ".csv"

	)

	evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])
