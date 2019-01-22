import logging

from skmultiflow.trees.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.gini_split_criterion import GiniSplitCriterion
from skmultiflow.trees import HoeffdingTree
from operator import attrgetter


GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'

# Logger
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



class HATT(HoeffdingTree):
    """ Hoeffding Anytime Tree for evolving data streams.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.

    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.

    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.

    split_criterion: string (default='info_gain')
        Split criterion to use.

        - 'gini' - Gini
        - 'info_gain' - Information Gain

    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.

    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.

    binary_split: boolean (default=False)
        If True, only allow binary splits.

    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.

    remove_poor_atts: boolean (default=False)
        If True, disable poor attributes.

    no_preprune: boolean (default=False)
        If True, disable pre-pruning.

    leaf_prediction: string (default='nba')
        Prediction mechanism used at leafs.

        - 'mc' - Majority Class
        - 'nb' - Naive Bayes
        - 'nba' - Naive Bayes Adaptive

    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.

    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    The Hoeffding Adaptive Tree [1]_ uses ADWIN [2]_ to monitor performance of branches on the tree and to replace them
    with new branches when their accuracy decreases if the new branches are more accurate.

    References
    ----------
    .. [1] Chaitanya Manapragada, Geoffrey I. Webb, and Mahsa Salehi. 2018.\
       Ex-tremely Fast Decision Tree. In Proceedings of ACM conference (KDD’18).\
       ACM, New York, NY, USA, Article 4, 9 pages.


    Examples
    --------
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.data import RandomTreeGenerator
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> from skmultiflow.trees.hoeffding_anytime_tree import HATT

    >>> stream = RandomTreeGenerator(
    >>>    tree_random_state=0,
    >>>    sample_random_state=0
    >>> )

    >>>stream.prepare_for_use()
    >>> h = [
    >>>    HoeffdingTree(),
    >>>    HATT()
    >>> ]

    >>> evaluator = EvaluatePrequential(
    >>>    pretrain_size=100,
    >>>    show_plot=True,
    >>>    max_samples=100000,
    >>>    metrics=['accuracy'],
    >>>    batch_size=1
    >>> )
    >>> evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])

    """
    # =============================================
    # == Hoeffding Anytime Tree implementation ====
    # =============================================

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None):
        super().__init__(max_byte_size, memory_estimate_period, grace_period,
         split_criterion, split_confidence, tie_threshold, binary_split,
          stop_mem_management, remove_poor_atts, no_preprune, leaf_prediction,
          nb_threshold, nominal_attributes)
        self.number_of_splits = 0
        self.number_of_resplits = 0
        self.number_of_unsplits = 0


    class HattSplitNode(HoeffdingTree.SplitNode):

        def __init__(self, learning_node, split_test, class_observations):
            self.learning_node = learning_node
            super().__init__(split_test, class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            self.learning_node.learn_from_instance(X, y, weight, ht)


    def _sort_instance_to_leaf(self, X):
        path = []
        current = self._tree_root
        parent = None
        branch = -1
        while True:
            path.append(HoeffdingTree.FoundNode(current, parent, branch))
            if isinstance(current, HoeffdingTree.SplitNode):
                child_index = current.instance_child_index(X)
                if child_index >= 0:
                    child = current.get_child(child_index)
                    if child is not None:
                        branch = child_index
                        parent = current
                        current = child
            else:
                break
        return path

    # Override HoeffdingTree
    def _partial_fit(self, X, y, weight):

        # initialize the tree
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()

        # sort the example into a leaf
        path = self._sort_instance_to_leaf(X)
        leaf_node = path[-1].node

        for found_node in path:
            node = found_node.node
            node.learn_from_instance(X, y, 1, self)
            if node == leaf_node:
                self._attempt_to_split(node, found_node.parent, found_node.parent_branch)
            else:
                self._re_evaluate_best_split(node, found_node.parent, found_node.parent_branch)


    def _re_evaluate_best_split(self, haat_node, parent, parent_idx):

        learning_node = haat_node.learning_node

        if self._split_criterion == GINI_SPLIT:
            split_criterion = GiniSplitCriterion()
        elif self._split_criterion == INFO_GAIN_SPLIT:
            split_criterion = InfoGainSplitCriterion()
        else:
            split_criterion = InfoGainSplitCriterion()

        best_split_suggestions = learning_node.get_best_split_suggestions(split_criterion, self)
        best_split_suggestions.sort(key=attrgetter('merit'))
        best_suggestion = best_split_suggestions[-1]

        hoeffding_bound = self.compute_hoeffding_bound(
            split_criterion.get_range_of_merit(learning_node.get_observed_class_distribution()),
            self.split_confidence, learning_node.get_weight_seen())

        current_split = None
        for split_suggestion in best_split_suggestions:
            if same_split(split_suggestion.split_test, haat_node._split_test):
                current_split = split_suggestion
                break

        if (best_suggestion.merit - current_split.merit > hoeffding_bound):
            if best_suggestion.split_test is None:
                # replace with a leaf, i.e. the learning node in this case
                self.number_of_unsplits += 1
                if parent is None:
                    self._tree_root = learning_node
                else:
                    parent.set_child(parent_idx, learning_node)
                return True
            elif current_split.split_test != best_suggestion.split_test:

                # replace with a node that splits on best_suggestion
                new_split = self.HattSplitNode(
                    learning_node,
                    best_suggestion.split_test,
                    learning_node.get_observed_class_distribution()
                )

                for i in range(best_suggestion.num_splits()):
                    new_child = self._new_learning_node(best_suggestion.resulting_class_distribution_from_split(i))
                    new_split.set_child(i, new_child)

                self.number_of_resplits += 1
                if parent is None:
                    self._tree_root = new_split
                else:
                    parent.set_child(parent_idx, new_split)
                return True

        return False

    # Override HoeffdingTree
    def _attempt_to_split(self, node, parent, parent_idx):

        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()

            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter('merit'))

            hoeffding_bound = self.compute_hoeffding_bound(
                split_criterion.get_range_of_merit(node.get_observed_class_distribution()),
                self.split_confidence, node.get_weight_seen())

            best_suggestion = best_split_suggestions[-1]

            no_split = None
            for split_suggestion in best_split_suggestions:
                if split_suggestion.split_test is None:
                    no_split = split_suggestion
                    break

            # split according to best_suggestion
            if (best_suggestion.merit - no_split.merit > hoeffding_bound and best_suggestion is not None):

                new_split = self.HattSplitNode(
                    node,
                    best_suggestion.split_test,
                    node.get_observed_class_distribution()
                )

                for i in range(best_suggestion.num_splits()):
                    new_child = self._new_learning_node(best_suggestion.resulting_class_distribution_from_split(i))
                    new_split.set_child(i, new_child)

                self.number_of_splits += 1
                if parent is None:
                    self._tree_root = new_split
                else:
                    parent.set_child(parent_idx, new_split)



def same_split(split_a, split_b):
    if split_a is None or split_b is None:
        return split_a is None and split_b is None
    return split_a._att_idx == split_b._att_idx
