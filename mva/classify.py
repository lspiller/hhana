# stdlib imports
import os
import pickle
from operator import itemgetter
import types
import shutil
from cStringIO import StringIO

# numpy imports
import numpy as np

# scikit-learn imports
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from prettytable import PrettyTable

# root_numpy imports
from root_numpy import rec2array, fill_hist

# local imports
from . import log; log = log[__name__]
from . import MMC_MASS, MMC_PT
from .plotting import plot_grid_scores
from . import variables, CACHE_DIR, BDT_DIR, CBA_DIR
from .systematics import systematic_name
from .grid_search import BoostGridSearchCV

import tempfile
import ROOT
from ROOT import TFile, TMVA, TCut
import math
from rootpy.tree import Cut
from root_numpy.tmva import (add_classification_events,
                             evaluate_method)


def significance( s, b ):
    log.info("Significance for n_sig, n_bkg = {}, {}".format(s,b))
    # Double because training sample is half luminosity
    if (2 * (s+b) * math.log( 1+ s/b ) - 2 * s) < 0:
        log.warning("Significiance <0 for s={}, b={}".format(s,b))
        return 0
    if b < 0:
        log.warning("b < 0 for s={}, b={}".format(s,b))
        return 0
    if s < 0:
        log.warning("s < 0 for s={}, b={}".format(s,b))
        return 0
    return math.sqrt( 2 * (s+b) * math.log( 1+ s/b ) - 2 * s )


def print_feature_ranking(clf, fields):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    log.info("Feature ranking:")
    out = StringIO()
    print >> out
    print >> out
    print >> out, r"\begin{tabular}{c|c|c}"
    table = PrettyTable(["Rank", "Variable", "Importance"])
    print >> out, r"\hline\hline"
    print >> out, r"Rank & Variable & Importance\\"
    for f, idx in enumerate(indices):
        table.add_row([f + 1,
            fields[idx],
            '%.3f' % importances[idx]])
        print >> out, r"%d & %s & %.3f\\" % (f + 1,
            variables.HH_VARIABLES[fields[idx]]['title'],
            importances[idx])
    print >> out, r"\end{tabular}"
    print >> out
    print >> out, table.get_string(hrules=1)
    log.info(out.getvalue())


def histogram_scores(hist_template, scores,
                     min_score=None, max_score=None,
                     inplace=False):
    if not inplace:
        hist = hist_template.Clone(name=hist_template.name + "_scores")
        hist.Reset()
    else:
        hist = hist_template
    if min_score is not None:
        log.info("cutting out scores below %f" % min_score)
    if max_score is not None:
        log.info("cutting out scores above %f" % max_score)
    if isinstance(scores, np.ndarray):
        if min_score is not None:
            scores = scores[scores > min_score]
        if max_score is not None:
            scores = scores[scores < max_score]
        fill_hist(hist, scores)
    elif isinstance(scores, tuple):
        # data
        scores, weight = scores
        if min_score is not None:
            scores_idx = scores > min_score
            scores = scores[scores_idx]
            weight = weight[scores_idx]
        if max_score is not None:
            scores_idx = scores < max_score
            scores = scores[scores_idx]
            weight = weight[scores_idx]
        assert (weight == 1).all()
        fill_hist(hist, scores)
    elif isinstance(scores, dict):
        # non-data with possible systematics
        # nominal case:
        nom_scores, nom_weight = scores['NOMINAL']
        if min_score is not None:
            scores_idx = nom_scores > min_score
            nom_scores = nom_scores[scores_idx]
            nom_weight = nom_weight[scores_idx]
        if max_score is not None:
            scores_idx = nom_scores < max_score
            nom_scores = nom_scores[scores_idx]
            nom_weight = nom_weight[scores_idx]
        fill_hist(hist, nom_scores, nom_weight)
        # systematics
        sys_hists = {}
        for sys_term, (sys_scores, sys_weight) in scores.items():
            if sys_term == 'NOMINAL':
                continue
            if min_score is not None:
                scores_idx = sys_scores > min_score
                sys_scores = sys_scores[scores_idx]
                sys_weight = sys_weight[scores_idx]
            if max_score is not None:
                scores_idx = sys_scores < max_score
                sys_scores = sys_scores[scores_idx]
                sys_weight = sys_weight[scores_idx]
            sys_hist = hist.Clone(
                name=hist.name + "_" + systematic_name(sys_term))
            sys_hist.Reset()
            fill_hist(sys_hist, sys_scores, sys_weight)
            sys_hists[sys_term] = sys_hist
        hist.systematics = sys_hists
    else:
        raise TypeError("scores not an np.array, tuple or dict")
    return hist


def write_score_hists(f, mass, scores_list, hist_template, no_neg_bins=True):
    sys_hists = {}
    for samp, scores_dict in scores_list:
        for sys_term, (scores, weights) in scores_dict.items():
            if sys_term == 'NOMINAL':
                suffix = ''
            else:
                suffix = '_' + '_'.join(sys_term)
            hist = hist_template.Clone(
                    name=samp.name + ('_{0}'.format(mass)) + suffix)
            fill_hist(hist, scores, weights)
            if sys_term not in sys_hists:
                sys_hists[sys_term] = []
            sys_hists[sys_term].append(hist)
    f.cd()
    for sys_term, hists in sys_hists.items():
        bad_bins = []
        if no_neg_bins:
            # check for negative bins over all systematics and zero them out
            # negative bins cause lots of problem in the limit setting
            # negative bin contents effectively means
            # the same as "no events here..."
            total_hist = sum(hists)
            for bin, content in enumerate(total_hist):
                if content < 0:
                    log.warning("Found negative bin %d (%f) for "
                                "systematic %s" % (
                                    bin, content, sys_term))
                    bad_bins.append(bin)
        for hist in hists:
            for bin in bad_bins:
                # zero out bad bins
                hist[bin] = 0.
            hist.Write()


def make_dataset(signals, backgrounds,
                 category, region, fields,
                 cuts=None):
    signal_arrs = []
    signal_weight_arrs = []
    background_arrs = []
    background_weight_arrs = []

    for signal in signals:
        rec = signal.merged_records(
            category=category,
            region=region,
            fields=fields,
            cuts=cuts)
        signal_weight_arrs.append(rec['weight'])
        signal_arrs.append(rec2array(rec, fields))

    for background in backgrounds:
        rec = background.merged_records(
            category=category,
            region=region,
            fields=fields,
            cuts=cuts)
        background_weight_arrs.append(rec['weight'])
        background_arrs.append(rec2array(rec, fields))

    signal_array = np.concatenate(signal_arrs)
    signal_weight_array = np.concatenate(signal_weight_arrs)
    background_array = np.concatenate(background_arrs)
    background_weight_array = np.concatenate(background_weight_arrs)

    return (signal_array, signal_weight_array,
            background_array, background_weight_array)


def make_partitioned_dataset(signals, backgrounds,
                             category, region, fields,
                             partition_key,
                             cuts=None):
    signal_arrs = []
    signal_weight_arrs = []
    background_arrs = []
    background_weight_arrs = []

    for signal in signals:
        left, right = signal.partitioned_records(
            category=category,
            region=region,
            fields=fields,
            cuts=cuts,
            key=partition_key)
        signal_weight_arrs.append(
            (left['weight'], right['weight']))
        signal_arrs.append(
            (rec2array(left, fields),
            rec2array(right, fields)))

    for background in backgrounds:
        left, right = background.partitioned_records(
            category=category,
            region=region,
            fields=fields,
            cuts=cuts,
            key=partition_key)
        background_weight_arrs.append(
            (left['weight'], right['weight']))
        background_arrs.append(
            (rec2array(left, fields),
            rec2array(right, fields)))

    return (signal_arrs, signal_weight_arrs,
            background_arrs, background_weight_arrs)


def get_partition(s, sw, b, bw, partition_idx):
    # select partition and merge arrays
    s = np.concatenate(map(itemgetter(partition_idx), s))
    sw = np.concatenate(map(itemgetter(partition_idx), sw))
    b = np.concatenate(map(itemgetter(partition_idx), b))
    bw = np.concatenate(map(itemgetter(partition_idx), bw))
    return s, sw, b, bw


def prepare_dataset(signal_train, signal_weight_train,
                    background_train, background_weight_train,
                    max_sig=None,
                    max_bkg=None,
                    norm_sig_to_bkg=True,
                    same_size_sig_bkg=True,
                    remove_negative_weights=False):
    if remove_negative_weights:
        # remove samples from the training sample with a negative weight
        signal_train = signal_train[signal_weight_train >= 0]
        background_train = background_train[background_weight_train >= 0]
        signal_weight_train = signal_weight_train[signal_weight_train >= 0]
        background_weight_train = background_weight_train[background_weight_train >= 0]
        log.info("removing events with negative weights")

    if max_sig is not None and max_sig < len(signal_train):
        subsample = np.random.permutation(len(signal_train))[:max_sig_train]
        signal_train = signal_train[subsample]
        signal_weight_train = signal_weight_train[subsample]
        log.info("signal stats reduced to user-specified maximum")

    if max_bkg is not None and max_bkg < len(background_train):
        subsample = np.random.permutation(len(background_train))[:max_bkg_train]
        background_train = background_train[subsample]
        background_weight_train = background_weight_train[subsample]
        log.info("background stats reduced to user-specified maximum")

    if same_size_sig_bkg:
        if len(background_train) > len(signal_train):
            # random subsample of background so it's the same size as signal
            subsample = np.random.permutation(
                len(background_train))[:len(signal_train)]
            background_train = background_train[subsample]
            background_weight_train = background_weight_train[subsample]
            log.info("number of background events reduced "
                     "to match number of signal events")
        elif len(background_train) < len(signal_train):
            # random subsample of signal so it's the same size as background
            subsample = np.random.permutation(
                len(signal_train))[:len(background_train)]
            signal_train = signal_train[subsample]
            signal_weight_train = signal_weight_train[subsample]
            log.info("number of signal events reduced "
                     "to match number of background events")

    if norm_sig_to_bkg:
        # normalize signal to background
        signal_weight_train *= (
            background_weight_train.sum() / signal_weight_train.sum())
        log.info("normalizing signal to match background")

    log.info("training Samples:")
    log.info("signal: %d events, %s features" % signal_train.shape)
    log.info("sum(signal weights): %f" % signal_weight_train.sum())
    log.info("background: %d events, %s features" % background_train.shape)
    log.info("sum(background weights): %f" % background_weight_train.sum())
    log.info("total: %d events" % (
        signal_train.shape[0] +
        background_train.shape[0]))

    sample_train = np.concatenate((background_train, signal_train))
    sample_weight_train = np.concatenate(
        (background_weight_train, signal_weight_train))
    labels_train = np.concatenate(
        (np.zeros(len(background_train)), np.ones(len(signal_train))))

    # random permutation of training sample
    perm = np.random.permutation(len(labels_train))
    sample_train = sample_train[perm]
    sample_weight_train = sample_weight_train[perm]
    labels_train = labels_train[perm]
    return sample_train, labels_train, sample_weight_train


class Classifier(object):
    # minimal list of spectators
    SPECTATORS = [
        MMC_PT,
        MMC_MASS,
    ]

    def __init__(self,
                 mass,
                 fields,
                 category,
                 region,
                 cuts=None,
                 spectators=None,
                 output_suffix="",
                 clf_output_suffix="",
                 partition_key='ditau_tau0_eta*100',#'EventNumber',
                 transform=True,
                 mmc=True):

        fields = fields[:]
        if not mmc:
            try:
                fields.remove(MMC_MASS)
            except ValueError:
                pass

        self.mass = mass
        self.fields = fields
        self.category = category
        self.region = region
        self.spectators = spectators
        self.output_suffix = output_suffix
        self.clf_output_suffix = clf_output_suffix
        self.partition_key = partition_key
        self.transform = transform
        self.mmc = mmc
        self.background_label = 0
        self.signal_label = 1

        if spectators is None:
            spectators = []

        # merge in minimal list of spectators
        for spec in Classifier.SPECTATORS:
            if spec not in spectators and spec not in fields:
                spectators.append(spec)

        self.all_fields = fields + spectators

        assert 'weight' not in fields

        # classifiers for the left and right partitions
        # each trained on the opposite partition
        self.clfs = None

    def binning(self, year, overflow=None, suffix=''):
        # get the binning (see the optimize-binning script)
        with open(os.path.join(CACHE_DIR, 'binning/binning_{0}_{1}_{2}_{3}.pickle'.format(
                               self.category.name, self.mass, year % 1000, suffix))) as f:
            binning = pickle.load(f)
        if overflow is not None:
            binning[0] -= overflow
            binning[-1] += overflow
        return binning

    def load(self, swap=False):
        """
        If swap is True then use the internal classifiers on the "wrong"
        partitions. This is used when demonstrating stability in data. The
        shape of the data distribution should be the same for both classifiers.
        """
        use_cache = True
        # attempt to load existing classifiers
        clfs = [None, None]
        for partition_idx in range(2):

            category_name = self.category.get_parent().name
            clf_filename = os.path.join(BDT_DIR,
                'clf_{0}_{1}{2}_{3}.pickle'.format(
                category_name, self.mass,
                self.clf_output_suffix, partition_idx))

            log.info("attempting to open %s ..." % clf_filename)
            if os.path.isfile(clf_filename):
                # use a previously trained classifier
                log.info("found existing classifier in %s" % clf_filename)
                with open(clf_filename, 'r') as f:
                    clf = pickle.load(f)
                out = StringIO()
                print >> out
                print >> out
                print >> out, clf
                log.info(out.getvalue())
                print_feature_ranking(clf, self.fields)
                if swap:
                    # DANGER
                    log.warning("will apply classifiers on swapped partitions")
                    clfs[partition_idx] = clf
                else:
                    clfs[(partition_idx + 1) % 2] = clf
            else:
                log.warning("could not open %s" % clf_filename)
                use_cache = False
                break
        if use_cache:
            self.clfs = clfs
            log.info("using previously trained classifiers")
            return True
        else:
            log.warning(
                "unable to load previously trained "
                "classifiers; train new ones")
            return False

    def train(self,
              signals,
              backgrounds,
              cuts=None,
              max_sig=None,
              max_bkg=None,
              norm_sig_to_bkg=True,
              same_size_sig_bkg=False,
              remove_negative_weights=False,
              max_trees=200,
              min_trees=1,
              learning_rate=0.1,
              max_fraction=0.3,
              min_fraction=0.001,
              min_fraction_steps=200,
              cv_nfold=10,
              n_jobs=-1,
              dry_run=False):
        """
        Determine best BDTs on left and right partitions. Each BDT will then be
        used on the other partition.
        """
        signal_arrs, signal_weight_arrs, \
        background_arrs, background_weight_arrs = make_partitioned_dataset(
            signals, backgrounds,
            category=self.category,
            region=self.region,
            fields=self.fields,
            cuts=cuts,
            partition_key=self.partition_key)

        if not dry_run:
            self.clfs = [None, None]

        for partition_idx in range(2):

            clf_filename = os.path.join(BDT_DIR,
                'clf_{0}_{1}{2}_{3}'.format(
                self.category.name, self.mass,
                self.clf_output_suffix, partition_idx))

            signal_train, signal_weight_train, \
            background_train, background_weight_train = get_partition(
                signal_arrs, signal_weight_arrs,
                background_arrs, background_weight_arrs,
                partition_idx)

            sample_train, labels_train, sample_weight_train = prepare_dataset(
                signal_train, signal_weight_train,
                background_train, background_weight_train,
                max_sig=max_sig,
                max_bkg=max_bkg,
                norm_sig_to_bkg=norm_sig_to_bkg,
                same_size_sig_bkg=same_size_sig_bkg,
                remove_negative_weights=remove_negative_weights)

            if dry_run:
                return

            log.info("training a new classifier...")

            if partition_idx == 0:

                # grid search params
                # min_samples_leaf
                #min_leaf_high = int((sample_train.shape[0] / 8) *
                #    (cv_nfold - 1.) / cv_nfold)
                #min_leaf_low = max(10, int(min_leaf_high / 100.))
                #min_leaf_step = max((min_leaf_high - min_leaf_low) / 100, 1)
                #min_samples_leaf = range(
                #    min_leaf_low, min_leaf_high, min_leaf_step)

                # min_fraction_leaf
                min_fraction_leaf = np.linspace(
                    min_fraction, max_fraction, min_fraction_steps)

                grid_params = {
                    #'base_estimator__min_samples_leaf': min_samples_leaf,
                    'base_estimator__min_fraction_leaf': min_fraction_leaf,
                }

                # create a BDT
                clf = AdaBoostClassifier(
                    DecisionTreeClassifier(),
                    learning_rate=learning_rate,
                    algorithm='SAMME.R',
                    random_state=0)

                # more efficient grid-search for boosting
                grid_clf = BoostGridSearchCV(
                    clf, grid_params,
                    max_n_estimators=max_trees,
                    min_n_estimators=min_trees,
                    #score_func=accuracy_score,
                    score_func=roc_auc_score, # area under the ROC curve
                    cv=StratifiedKFold(labels_train, cv_nfold),
                    n_jobs=n_jobs)

                #grid_clf = GridSearchCV(
                #    clf, grid_params,
                #    score_func=accuracy_score,
                #    cv = StratifiedKFold(labels_train, cv_nfold),
                #    n_jobs=n_jobs)

                log.info("")
                log.info("using a %d-fold cross validation" % cv_nfold)
                log.info("performing a grid search over these parameter values:")
                for param, values in grid_params.items():
                    log.info('{0} {1}'.format(param.split('__')[-1], values))
                log.info("Minimum number of trees: %d" % min_trees)
                log.info("Maximum number of trees: %d" % max_trees)
                log.info("")
                log.info("training new classifiers ...")

                # perform the cross-validated grid-search
                grid_clf.fit(
                    sample_train, labels_train,
                    sample_weight=sample_weight_train)

                clf = grid_clf.best_estimator_
                grid_scores = grid_clf.grid_scores_

                log.info("Best score: %f" % grid_clf.best_score_)
                log.info("Best Parameters:")
                log.info(grid_clf.best_params_)

                # plot a grid of the scores
                plot_grid_scores(
                    grid_scores,
                    best_point={
                        'base_estimator__min_fraction_leaf':
                        clf.base_estimator.min_fraction_leaf,
                        'n_estimators':
                        clf.n_estimators},
                    params={
                        'base_estimator__min_fraction_leaf':
                        'leaf fraction',
                        'n_estimators':
                        'trees'},
                    name=(self.category.name +
                          ("_{0}".format(self.mass)) +
                          self.output_suffix +
                          ("_{0}".format(partition_idx))))

                # save grid scores
                with open('{0}_grid_scores.pickle'.format(clf_filename), 'w') as f:
                    pickle.dump(grid_scores, f)

                # scale up the min-leaf and retrain on the whole set
                #min_samples_leaf = clf.base_estimator.min_samples_leaf
                #clf = sklearn.clone(clf)
                #clf.base_estimator.min_samples_leaf = int(
                #    min_samples_leaf *
                #        cv_nfold / float(cv_nfold - 1))
                #clf.fit(sample_train, labels_train,
                #        sample_weight=sample_weight_train)
                #log.info("After scaling up min_leaf")
                #out = StringIO()
                #print >> out
                #print >> out
                #print >> out, clf
                #log.info(out.getvalue())

            else: # training on the other partition
                log.info("training a new classifier ...")

                # use same params as in first partition
                clf = sklearn.clone(clf)
                out = StringIO()
                print >> out
                print >> out
                print >> out, clf
                log.info(out.getvalue())

                clf.fit(sample_train, labels_train,
                        sample_weight=sample_weight_train)

            # export to graphviz dot format
            if os.path.isdir(clf_filename):
                shutil.rmtree(clf_filename)
            os.mkdir(clf_filename)
            for itree, tree in enumerate(clf):
                export_graphviz(
                    tree,
                    out_file=os.path.join(
                        clf_filename,
                        'tree_{0:04d}.dot'.format(itree)),
                    feature_names=self.all_fields)

            with open('{0}.pickle'.format(clf_filename), 'w') as f:
                pickle.dump(clf, f)

            print_feature_ranking(clf, self.fields)

            self.clfs[(partition_idx + 1) % 2] = clf

    def classify(self, sample, category, region,
                 cuts=None, systematic='NOMINAL'):

        if self.clfs == None:
            raise RuntimeError("you must train the classifiers first")

        partitions = sample.partitioned_records(
            category=category,
            region=region,
            fields=self.fields,
            cuts=cuts,
            systematic=systematic,
            num_partitions=2,
            return_idx=True,
            key=self.partition_key)

        score_idx = [[], []]
        for i, partition in enumerate(partitions):
            for rec, idx in partition:
                weight = rec['weight']
                arr = rec2array(rec, self.fields)
                # each classifier is never used on the partition that trained it
                scores = self.clfs[i].decision_function(arr)
                score_idx[i].append((idx, scores, weight))

        # must preserve order of scores wrt the other fields!
        # merge the scores and weights according to the idx
        merged_scores = []
        merged_weight = []
        for left, right in zip(*score_idx):
            left_idx, left_scores, left_weight = left
            right_idx, right_scores, right_weight = right
            insert_idx = np.searchsorted(left_idx, right_idx)
            scores = np.insert(left_scores, insert_idx, right_scores)
            weight = np.insert(left_weight, insert_idx, right_weight)
            merged_scores.append(scores)
            merged_weight.append(weight)

        scores = np.concatenate(merged_scores)
        weight = np.concatenate(merged_weight)

        if self.transform:
            log.info("classifier scores are transformed")
            if isinstance(self.transform, types.FunctionType):
                # user-defined transformation
                scores = self.transform(scores)
            else:
                # logistic tranformation used by TMVA (MethodBDT.cxx)
                scores = -1 + 2.0 / (1.0 +
                    np.exp(-self.clfs[0].n_estimators *
                            self.clfs[0].learning_rate * scores / 1.5))

        return scores, weight

class CBA(object):

    def __init__(self,
                 mass,
                 cut_fields,
                 fischer_fields,
                 category,
                 region,
                 cuts=None,
                 spectators=None,
                 output_suffix="",
                 clf_output_suffix="",
                 partition_key='ditau_tau0_eta*100',#'EventNumber',
                 transform=True,
                 mmc=True):


        cut_fields = cut_fields[:]
        fischer_fields = fischer_fields[:]

        self.mass = mass
        self.cut_fields = cut_fields
        self.fischer_fields = fischer_fields
        self.category = category
        self.region = region
        self.spectators = spectators
        self.output_suffix = output_suffix
        self.clf_output_suffix = clf_output_suffix
        self.partition_key = partition_key
        self.transform = transform
        self.mmc = mmc
        self.background_label = 0
        self.signal_label = 1

        self.fischer_expression = ''
        self.opti_cuts = []
        self.cut_arrows = {}

        category_name = self.category.get_parent().name
        self.cut_filename = os.path.join(CBA_DIR,
            'cba_{0}_{1}_{2}.pickle'.format(
            category_name, self.mass,
            self.clf_output_suffix))
        self.fischer_filename = os.path.join(CBA_DIR,
            'fischer_{0}_{1}_{2}.pickle'.format(
            category_name, self.mass,
            self.clf_output_suffix))

        if spectators is None:
            spectators = []

        # merge in minimal list of spectators
        for spec in Classifier.SPECTATORS:
            if spec not in spectators and (spec not in cut_fields or spec not in fischer_fields):
                spectators.append(spec)

        self.all_fields = fischer_fields + cut_fields + spectators

        assert 'weight' not in cut_fields
        assert 'weight' not in fischer_fields
        self.cba = [Cut('1'), Cut('1')]

    def load(self):
        """
        If swap is True then use the internal classifiers on the "wrong"
        partitions. This is used when demonstrating stability in data. The
        shape of the data distribution should be the same for both classifiers.
        """
        # attempt to load existing classifiers
        log.info("attempting to open %s ..." % self.cut_filename)
        if os.path.isfile(self.cut_filename):
            # use a previously trained classifier
            log.info("found existing classifier in %s" % self.cut_filename)
            with open(self.cut_filename, 'r') as f:
                cut = pickle.load(f)
            out = StringIO()
            print >> out
            print >> out
            print >> out, cut
            log.info(out.getvalue())
        else:
            log.warning("could not open %s" % self.cut_filename)
            cut = Cut('1')
        self.cba[0] = cut

        if os.path.isfile(self.fischer_filename):
            # use a previously trained classifier
            log.info("found existing classifier in %s" % self.cut_filename)
            with open(self.fischer_filename, 'r') as f:
                fischer = pickle.load(f)
            out = StringIO()
            print >> out
            print >> out
            print >> out, fischer
            log.info(out.getvalue())
        else:
            log.warning("could not open %s" % self.fischer_filename)
        self.cba[1] = fischer

    def format_cut( self, lo, hi ):
        if   lo > -1E30 and hi < 1E30: return '{0:f} < {1} <= {2:f}'
        elif hi <  1E30:             return '{1} <= {2:f}'
        elif lo > -1E30:             return '{0:f} < {1}'
        else: '1'


    def format_fischer( self, coeff, varnames ):
        for c, v in zip( coeff, varnames+[1] ):
            terms = '{} * {}'.format( c, v ) if v != 1 else '{}'.format( c )
        expression =  ('+'.join(terms) + ' >= 0')
        return expression

    def train_fischer(self,
              signals,
              backgrounds,
              cuts=None,
              norm_sig_to_bkg=False,
              same_size_sig_bkg=False,
              max_sig=None,
              max_bkg=None,
              remove_negative_weights=False):
        """
        Train with partition_idx = 0
        Find optimal working point on partition_idx = 1
        """
        signal_arrs, signal_weight_arrs, \
        background_arrs, background_weight_arrs = make_partitioned_dataset(
            signals, backgrounds,
            category=self.category,
            region=self.region,
            fields=self.fischer_fields,
            cuts=cuts,
            partition_key=self.partition_key)


        log.info("Optimising cuts for cba analysis")

        tmpdir = tempfile.mkdtemp()
        output = TFile(
                os.path.join(tmpdir, 'tmva_output.root'), 'recreate')


        log.info("Computing optimal cuts for: {}".format(self.category.name))

        factory = TMVA.Factory('MyFischer', output,
                               'AnalysisType=Classification:V')
        config = TMVA.gConfig()
        config.GetIONames().fWeightFileDir = tmpdir
        # Change self.fields to cuts and fischer variables separately
        for var in self.fischer_fields:
            factory.AddVariable(var, 'F')


        for partition_idx in range(2):

            signal_train, signal_weight_train, \
            background_train, background_weight_train = get_partition(
                signal_arrs, signal_weight_arrs,
                background_arrs, background_weight_arrs,
                partition_idx)

            sample_train, labels_train, sample_weight_train = prepare_dataset(
                signal_train, signal_weight_train,
                background_train, background_weight_train,
                max_sig=max_sig,
                max_bkg=max_bkg,
                norm_sig_to_bkg=norm_sig_to_bkg,
                same_size_sig_bkg=same_size_sig_bkg,
                remove_negative_weights=remove_negative_weights)

            # Time for some stupid TMVA things...
            # Get index of first signal event
            first_signal = np.nonzero(labels_train == 1)[0][0]
            # Swap this with first event
            sample_train[0], sample_train[first_signal] = \
                    sample_train[first_signal].copy(), sample_train[0].copy()
            labels_train[0], labels_train[first_signal] = \
                    labels_train[first_signal], labels_train[0]
            sample_weight_train[0], sample_weight_train[first_signal] = \
                    sample_weight_train[first_signal], sample_weight_train[0]

            # Call root_numpy's utility functions to add events from the arrays
            add_classification_events(factory,
                    sample_train, labels_train,
                    weights=sample_weight_train,
                    signal_label=1, test=partition_idx)

        factory.PrepareTrainingAndTestTree(TCut('1'), 'NormMode=EqualNumEvents')

        # Train a classifier
        factory.BookMethod(TMVA.Types.kLD, 'MyFischer', '')
        factory.TrainAllMethods()

        # Analyse cuts to find optimal working point
        fischercut = factory.GetMethod('MyFischer')

        coeff = fischercut.GetRegressionValues()
        fischer_cutstring = self.format_fischer( coeff, self.fischer_fields )
        log.info( 'fischer constraint: {}'.format(fischer_cutstring) )

        self.fischer_equation = coeff
        self.fischer_expression = fischer_cutstring
        self.cba[1] = Cut(fischer_cutstring)

        with open('{0}.pickle'.format(self.fischer_filename), 'w') as f:
            pickle.dump(Cut(fischer_cutstring), f)


    def train_cuts(self,
              signals,
              backgrounds,
              cuts=None,
              norm_sig_to_bkg=False,
              same_size_sig_bkg=False,
              max_sig=None,
              max_bkg=None,
              remove_negative_weights=False):
        """
        Train with partition_idx = 0
        Find optimal working point on partition_idx = 1
        """
        signal_arrs, signal_weight_arrs, \
        background_arrs, background_weight_arrs = make_partitioned_dataset(
            signals, backgrounds,
            category=self.category,
            region=self.region,
            fields=self.cut_fields,
            cuts=cuts,
            partition_key=self.partition_key)


        log.info("Optimising cuts for cba analysis")

        tmpdir = tempfile.mkdtemp()
        output = TFile(
                os.path.join(tmpdir, 'tmva_output.root'), 'recreate')


        log.info("Computing optimal cuts for: {}".format(self.category.name))

        factory = TMVA.Factory('MyCuts', output,
                               'AnalysisType=Classification:V')
        config = TMVA.gConfig()
        config.GetIONames().fWeightFileDir = tmpdir

        for var in self.cut_fields:
            factory.AddVariable(var, 'F')


        for partition_idx in range(2):

            signal_train, signal_weight_train, \
            background_train, background_weight_train = get_partition(
                signal_arrs, signal_weight_arrs,
                background_arrs, background_weight_arrs,
                partition_idx)

            sample_train, labels_train, sample_weight_train = prepare_dataset(
                signal_train, signal_weight_train,
                background_train, background_weight_train,
                max_sig=max_sig,
                max_bkg=max_bkg,
                norm_sig_to_bkg=norm_sig_to_bkg,
                same_size_sig_bkg=same_size_sig_bkg,
                remove_negative_weights=remove_negative_weights)

            # Time for some stupid TMVA things...
            # Get index of first signal event
            first_signal = np.nonzero(labels_train == 1)[0][0]
            # Swap this with first event
            sample_train[0], sample_train[first_signal] = \
                    sample_train[first_signal].copy(), sample_train[0].copy()
            labels_train[0], labels_train[first_signal] = \
                    labels_train[first_signal], labels_train[0]
            sample_weight_train[0], sample_weight_train[first_signal] = \
                    sample_weight_train[first_signal], sample_weight_train[0]

            # Call root_numpy's utility functions to add events from the arrays
            add_classification_events(factory,
                    sample_train, labels_train,
                    weights=sample_weight_train,
                    signal_label=1, test=partition_idx)

        factory.PrepareTrainingAndTestTree(TCut('1'), 'NormMode=EqualNumEvents')

        # Train a classifier
        arguments = {
                'FitMethod':'GA',
                'EffSel':True,
        # 'SampleSize':100,
                'VarProp':'FSmart'
                }
        options = []
        for param, value in arguments.items():
            if value is True:
                options.append(param)
            elif value is False:
                options.append('!{0}'.format(param))
            else:
                options.append('{0}={1}'.format(param, value))
        options = ':'.join(options)
        factory.BookMethod(TMVA.Types.kCuts, 'MyCuts', options)
        factory.TrainAllMethods()

        # Analyse cuts to find optimal working point
        methodcut = factory.GetMethod('MyCuts')

        d_vector = ROOT.std.vector('Double_t')
        best_eff, best_merit = 0., 0.

        efficiencies = np.arange(0.05, 0.95, 0.05)
        for eff in efficiencies:
            # Using trick here!
            # the sample_train,... would be the set denoted for training
            test_output = evaluate_method( methodcut, sample_train, eff )
            n0_mask = (test_output == 1)
            bkg_mask = (test_output == 1) & (labels_train == 0)
            sig_mask = (test_output == 1) & (labels_train ==1)
            n0 = np.sum(sample_weight_train[ n0_mask ])
            nbkg = np.sum(sample_weight_train[ bkg_mask ])
            nsig = np.sum(sample_weight_train[ sig_mask ])
            # training set = 1/2 sample. need to double expected events
            merit = significance( s=2*nsig, b=2*nbkg )

            log.info(('@ sig_eff = {}:\n'
                      '\tn0 = {},\n'
                      '\tsig = {},\n'
                      '\tbkg = {},\n'
                      '\tf.o.m = {}').format( eff, n0, nsig, nbkg, merit ))
            if best_merit < merit:
                best_eff, best_merit = eff, merit

        log.info(('\nOptimal cut location:\n'
                  '\teff = {}\n'
                  '\tf.o.m = {}').format( best_eff, best_merit ) )
        cutmin = d_vector()
        cutmax = d_vector()
        methodcut.GetCuts( best_eff, cutmin, cutmax )

        cuts = [
                self.format_cut(lo,hi).format(lo, c, hi)
                for c, lo, hi in zip( self.cut_fields, cutmin, cutmax )
            ]
        total_cuts = None
        log.info("Cuts at optimal position:\n\t{}".format(',\n\t'.join(cuts)))
        for c in cuts: total_cuts = (total_cuts & Cut(c)) if total_cuts else Cut(c)

        for c,lo,hi in zip(self.cut_fields, cutmin, cutmax):
            if lo > -1E29 and hi < 1E29:
                self.cut_arrows[c] = (lo,hi)
            elif lo > -1E29:
                self.cut_arrows[c] = (lo,)
            elif hi < 1E29:
                self.cut_arrows[c] = (hi,)

        self.opti_cut = cuts
        self.cba[0] = total_cuts

        with open('{0}.pickle'.format(self.cut_filename), 'w') as f:
            pickle.dump(total_cuts, f)

