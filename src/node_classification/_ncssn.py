from src.node_classification._node_learner import NodeLearner

import cvxpy as cp
import numpy as np
import scipy.sparse as sps
import itertools


class NCSSN(NodeLearner):
    def __init__(self, num_classes=2, verbosity=0, save_intermediate=None, independent_weight=1, dependent_weight=0.5,
                 classifier_weight=0.5, regularization_weight=1e-2, preference_size=1):
        self.independent_weight = independent_weight
        self.dependent_weight = dependent_weight
        self.classifier_weight = classifier_weight
        self.regularizatin_weight = regularization_weight
        self.preference_size = preference_size

        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        num_nodes = graph.num_nodes

        w_pos = graph.weights.maximum(0)
        w_neg = -graph.weights.minimum(0)

        w_pos.eliminate_zeros()
        w_neg.eliminate_zeros()

        w_pos_coo = graph.w_pos.tocoo()
        w_neg_coo = graph.w_neg.tocoo()
        # M = np.zeros()

        S = []
        for i in range(num_nodes):
            s_ = [item for item in itertools.product([i], w_pos.getrow(i).indices, w_neg.getrow(i).indices)]
            S += s_

        y = np.zeros((num_nodes, self.num_classes))
        y[labels['i'], labels['k']] = 1
        C = sps.diags([1 if i in labels['i'] else 0 for i in range(num_nodes)])
        S = np.array(S)

        w_pos_param = cp.Parameter(shape=w_pos.shape,value=w_pos)
        w_neg_param = cp.Parameter(shape=w_neg.shape,value=w_neg)

        h_p = cp.Variable((self.preference_size, self.preference_size), nonneg=True, name='h_p')
        h_n = cp.Variable((self.preference_size, self.preference_size), nonneg=True, name='h_n')
        u = cp.Variable((num_nodes, self.preference_size), nonneg=True, name='u')
        w = cp.Variable((self.preference_size, self.num_classes), name='w')

        expr_ind_pos = cp.sum_squares(w_pos_param - u @ u.T)
        expr_ind_neg = self.independent_weight * cp.power(cp.norm(w_neg - u @ h_n @ u.T), 2)
        expr_classifier = self.classifier_weight * cp.power(cp.norm(C @ (u @ w - y)), 2)
        expr_dep = self.dependent_weight * cp.sum(cp.maximum(0,
                                     cp.power(cp.norm(u[S[:, 0]] - u[S[:, 1]], axis=1), 2) -
                                     cp.power(cp.norm(u[S[:, 0]] - u[S[:, 2]], axis=1), 2)))
        expr_regularization = self.regularizatin_weight * (cp.power(cp.norm(h_p),2) +
                                                           cp.power(cp.norm(h_n),2) +
                                                           cp.power(cp.norm(u),2) +
                                                           cp.power(cp.norm(w),2))

        obj = cp.Minimize(
                          expr_ind_pos
                          # + expr_ind_neg
                          # + expr_classifier
                          # + expr_dep
                          # + expr_regularization
                          )

        problem = cp.Problem(objective=obj)
        solution = problem.solve()



        pass
