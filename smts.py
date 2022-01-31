import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import zip_longest
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier


def get_leaves_index(tree):
    '''
    Function reused (and lightly modified) from:
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    It returns all tree's leaves index.
    '''
    leaves_index = []
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have
        # a split node
        is_split_node = children_left[node_id] != children_right[node_id]

        # If a split node, append left and right children and depth to
        # `stack` so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            leaves_index.append(node_id)

    return sorted(leaves_index)


def turn_serie_into_symbolic(
        df,
        serie_id,
        trained_forest,
        translation_table,
        id_col_name='id',
        class_col_name='classname'):
    symbolic_data = {}
    serie = df[df[id_col_name] == serie_id]
    classname = serie[class_col_name].iloc[0] if class_col_name in serie \
        else None

    for estimator_index in range(
            0, len(trained_forest.estimators_)):
        tree = trained_forest.estimators_[estimator_index].tree_
        tree_translation_table = translation_table[estimator_index]

        repeated_leaves_indexes = trained_forest\
            .estimators_[estimator_index]\
            .apply(serie.drop([id_col_name, class_col_name],
                              errors='ignore', axis=1).to_numpy())
        counter = Counter(repeated_leaves_indexes)

        for leave_index in get_leaves_index(tree):
            symbolic_data[tree_translation_table[leave_index]] =\
                counter[leave_index]/serie.shape[0]

    return {'serie_id': serie_id,
            'classname': classname,
            'symbolic_data': symbolic_data}


class SMTS():
    '''
    It defines a Symbolic Multivariate Time Series classificator
    based on RandomForest and Symbolic Representation combination
    explained in M.Baydogan's paper:
    Learning a Symbolic Representation For Multivariate Time Series.
    https://doi.org/10.1007/s10618-014-0349-y
    '''

    def __init__(
            self,
            j_ins=150,
            n_symbols=5,
            j_ts=100,
            random_state=None,
            oob_score=True,
            n_jobs=-1):
        self.j_ins = j_ins
        self.n_symbols = n_symbols
        self.j_ts = j_ts
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score

        if self.n_symbols <= 1:
            raise Exception('n_symbols must be greater than 1')

        self.__trained_forest = RandomForestClassifier(
            n_estimators=self.j_ins,
            max_leaf_nodes=self.n_symbols,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.__symbolic_forest = RandomForestClassifier(
            n_estimators=self.j_ts,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=self.oob_score
        )

    def fit(self, X, y, id_col_name='id'):
        self.__clean_params()
        self.__trained_forest.fit(
            X.drop(id_col_name, axis=1), y)

        self.X_train_, self.y_train_ =\
            self.__into_symbolic_data(X.assign(classname=y), id_col_name)

        self.__symbolic_forest.fit(self.X_train_, self.y_train_)
        self.classes_ = self.__symbolic_forest.classes_
        self.oob_score_ = self.__symbolic_forest.oob_score_\
            if self.oob_score else None

    def score(self, X, y, id_col_name='id'):
        X_test, y_test =\
            self.__into_symbolic_data(X.assign(classname=y), id_col_name)
        return self.__symbolic_forest.score(X_test, y_test)

    def predict(self, X, id_col_name='id'):
        X_pred, _ = self.__into_symbolic_data(X, id_col_name)
        return self.__symbolic_forest.predict(X_pred)

    def clone(self):
        return SMTS(
            j_ins=self.j_ins,
            n_symbols=self.n_symbols,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

    def __clean_params(self):
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def __get_forest_translation_tables(self):
        forest_translation_table = {}
        symbols = range(0, self.n_symbols)

        for estimator_index in range(
                0, len(self.__trained_forest.estimators_)):
            tree = self.__trained_forest.estimators_[estimator_index].tree_
            tree_translation_table = {}

            for leave_index, symbol in zip_longest(
                    get_leaves_index(tree), symbols):
                tree_translation_table[leave_index] =\
                    str(estimator_index) + str(symbol)

            forest_translation_table[estimator_index] = tree_translation_table

        return forest_translation_table

    def __into_symbolic_data(
            self, df, id_col_name='id', class_col_name='classname'):
        translation_table = self.__get_forest_translation_tables()
        symbolic_data = defaultdict(list)
        classes = []

        series_symbolic_data = Parallel(n_jobs=self.n_jobs)(
            delayed(turn_serie_into_symbolic)(
                df,
                serie_id,
                self.__trained_forest,
                translation_table,
                id_col_name,
                class_col_name
            ) for serie_id in pd.unique(df[id_col_name])
        )
        symbolic_data = pd\
            .DataFrame([serie_symbolic_data['symbolic_data']
                        for serie_symbolic_data in series_symbolic_data])
        classes = np.asarray([serie_symbolic_data['classname']
                              for serie_symbolic_data in series_symbolic_data])

        return symbolic_data, classes
