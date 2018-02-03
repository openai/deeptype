cimport cython
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.algorithm cimport sort as stdsort
from multiprocessing import cpu_count, Queue
from libcpp.unordered_set cimport unordered_set

from threading import Thread

import time
import random
import numpy as np
cimport numpy as np
from .progressbar import get_progress_bar

from deap import (
    algorithms as deap_algorithms,
    base as deap_base,
    creator as deap_creator,
    tools as deap_tools
)


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


def greedy_disambiguate(tags):
    greedy_correct = 0
    total = 0
    remainder = []
    for dest, other_dest, times_pointed in tags:
        total += 1
        if len(other_dest) == 1 and dest == other_dest[0]:
            greedy_correct += 1
        elif other_dest[np.argmax(times_pointed)] == dest:
            greedy_correct += 1
        else:
            remainder.append((dest, other_dest, times_pointed))
    return greedy_correct, total, remainder


cdef bool* bool_ptr(array):
    return <bool*>(<long>array.ctypes.data)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int fast_disambiguate(const vector[int]& dest_vec,
                           const vector[vector[int]]& tags_vec,
                           const vector[vector[int]]& times_pointed_vec,
                           bool* all_classifications_ptr,
                           long rows,
                           long cols,
                           vector[int]& dims) nogil:
    cdef int correct = 0
    cdef int i = 0
    cdef int j = 0
    cdef int dim_idx = 0
    cdef int matches = 0
    cdef int max_match = 0
    cdef int max_match_val = 0
    cdef bool same_types

    for i in range(tags_vec.size()):
        matches = tags_vec[i].size()
        max_match_val = 0
        max_match = -1

        for j in range(tags_vec[i].size()):
            same_types = True
            for dim_idx in range(dims.size()):
                if all_classifications_ptr[dims[dim_idx] * cols + tags_vec[i][j]] != all_classifications_ptr[dims[dim_idx] * cols + dest_vec[i]]:
                    same_types = False
                    matches -= 1
                    break
            if same_types and times_pointed_vec[i][j] > max_match_val:
                max_match_val = times_pointed_vec[i][j]
                max_match = tags_vec[i][j]
        if matches == 1 or max_match == dest_vec[i]:
            correct += 1
    return correct


cdef struct BeamCandidate:
    double sum_auc
    double secondary_objective
    double objective
    bool stopped
    vector[int] kept_indices
    unordered_set[int] kept_indices_set



cdef inline BeamCandidate beam_candidate(double sum_auc,
                                         double secondary_objective,
                                         double objective,
                                         bool stopped,
                                         vector[int]& kept_indices,
                                         unordered_set[int]& kept_indices_set) nogil:
    cdef BeamCandidate cand
    cand.sum_auc = sum_auc
    cand.secondary_objective = secondary_objective
    cand.objective = objective
    cand.stopped = stopped
    cand.kept_indices = kept_indices
    cand.kept_indices_set = kept_indices_set
    return cand


cdef bool candidate_order(const BeamCandidate& left, const BeamCandidate& right) nogil:
    return left.objective > right.objective


cdef class BeamSearch:
    cdef vector[BeamCandidate] candidates
    cdef vector[vector[BeamCandidate]] new_candidates
    cdef long all_classifications_ptr
    cdef vector[int] all_ids
    cdef vector[float] all_aucs
    cdef vector[int] dest_vec
    cdef vector[int] subset_indices
    cdef vector[float] subset_aucs
    cdef vector[vector[int]] tags_vec
    cdef vector[vector[int]] times_pointed_vec
    cdef long rows
    cdef long cols
    cdef int num_workers

    def __init__(self,
                 BeamCandidate initial,
                 const vector[int]& dest_vec,
                 const vector[vector[int]]& tags_vec,
                 const vector[vector[int]]& times_pointed_vec,
                 long all_classifications_ptr,
                 long rows, long cols,
                 const vector[int]& all_ids,
                 const vector[float]& all_aucs,
                 int num_workers):

        self.candidates.push_back(initial)

        self.dest_vec = dest_vec
        self.tags_vec = tags_vec
        self.times_pointed_vec = times_pointed_vec
        self.all_classifications_ptr = all_classifications_ptr
        self.rows = rows
        self.cols = cols
        self.all_ids = all_ids
        self.all_aucs = all_aucs
        self.num_workers = num_workers

        # create worker output vectors here:
        for i in range(self.num_workers):
            self.new_candidates.push_back([])


@cython.boundscheck(False)
@cython.wraparound(False)
def beam_project_worker(int worker_idx,
                        int num_workers,
                        float penalty,
                        int greedy_correct_c,
                        double total_c,
                        int work_size,
                        int beam_width,
                        np.ndarray[int] total_work,
                        int iteration,
                        int subset,
                        BeamSearch beam_search,
                        pbar):
    cdef int i = 0
    cdef int beam_idx = 0
    cdef double improvement = 0
    cdef double objective = 0
    cdef vector[int] proposal
    cdef unordered_set[int] proposal_set
    cdef bool* all_classifications_ptr = <bool*>beam_search.all_classifications_ptr
    cdef long rows = beam_search.rows
    cdef long cols = beam_search.cols
    cdef int[:] total_work_view = total_work
    cdef int work_done = 0
    cdef bool use_subset = iteration > 0 and subset > 0
    cdef float auc
    cdef int index

    with nogil:
        beam_search.new_candidates[worker_idx].clear()
        for beam_idx in range(beam_search.candidates.size()):
            if not beam_search.candidates[beam_idx].stopped:
                for i in range(work_size):
                    if i % num_workers == worker_idx:
                        if use_subset:
                            index = beam_search.subset_indices[i]
                            auc = beam_search.subset_aucs[i]
                        else:
                            index = beam_search.all_ids[i]
                            auc = beam_search.all_aucs[i]

                        if beam_search.candidates[beam_idx].kept_indices_set.find(index) == beam_search.candidates[beam_idx].kept_indices_set.end():
                            proposal = beam_search.candidates[beam_idx].kept_indices
                            proposal.push_back(index)
                            proposal_set = beam_search.candidates[beam_idx].kept_indices_set
                            proposal_set.insert(index)
                            improvement = fast_disambiguate(
                                beam_search.dest_vec,
                                beam_search.tags_vec,
                                beam_search.times_pointed_vec,
                                all_classifications_ptr,
                                rows,
                                cols,
                                proposal
                            )
                            if proposal.size() == 0:
                                objective = greedy_correct_c
                            else:
                                objective = (
                                    (greedy_correct_c +
                                     improvement * ((beam_search.candidates[beam_idx].sum_auc +
                                                     auc) / proposal.size())) / total_c -
                                    # number of items is penalized
                                    proposal.size() * penalty
                                )
                            beam_search.new_candidates[worker_idx].push_back(beam_candidate(
                                beam_search.candidates[beam_idx].sum_auc + auc,
                                (greedy_correct_c + improvement) / total_c,
                                objective,
                                False,
                                proposal,
                                proposal_set
                            ))
                        work_done += 1
                        if work_done % 100 == 0:
                            total_work_view[worker_idx] = work_done
                            if worker_idx == 0:
                                with gil:
                                    pbar.update(total_work.sum())

            beam_search.new_candidates[worker_idx].push_back(
                beam_candidate(beam_search.candidates[beam_idx].sum_auc,
                               beam_search.candidates[beam_idx].secondary_objective,
                               beam_search.candidates[beam_idx].objective,
                               True,
                               beam_search.candidates[beam_idx].kept_indices,
                               beam_search.candidates[beam_idx].kept_indices_set))
        stdsort(beam_search.new_candidates[worker_idx].begin(), beam_search.new_candidates[worker_idx].end(), &candidate_order)
        if iteration == 0 and subset > 0:
            beam_search.new_candidates[worker_idx].erase(
                beam_search.new_candidates[worker_idx].begin() + subset,
                beam_search.new_candidates[worker_idx].end())
        else:
            beam_search.new_candidates[worker_idx].erase(
                beam_search.new_candidates[worker_idx].begin() + beam_width,
                beam_search.new_candidates[worker_idx].end())


@cython.boundscheck(False)
@cython.wraparound(False)
def beam_project(cached_satisfy, key2row, tags, aucs, ids,
                 float penalty, int beam_width, int subset=-1,
                 log=None):
    if beam_width < 1:
        raise ValueError("beam_width must be greater than 0.")
    all_classifications = None
    greedy_correct, total, remainder_tags = greedy_disambiguate(tags)
    cdef int greedy_correct_c = greedy_correct
    cdef double total_c = total
    cdef int current_best = greedy_correct
    cdef vector[vector[int]] tags_vec
    cdef vector[vector[int]] times_pointed_vec
    cdef vector[int] dest_vec
    cdef long rows = cached_satisfy.shape[0]
    cdef long cols = cached_satisfy.shape[1]
    for dest, other_dest, times_pointed in remainder_tags:
        tags_vec.push_back(other_dest)
        dest_vec.push_back(dest)
        times_pointed_vec.push_back(times_pointed)
    cdef bool* all_classifications_ptr = bool_ptr(cached_satisfy)
    cdef vector[int] all_ids = [key2row[(qid, relation)] for (qid, relation) in sorted(aucs.keys(), key=lambda x: aucs[x], reverse=True)]
    cdef vector[float] all_aucs = [aucs[key] for key in sorted(aucs.keys(), key=lambda x: aucs[x], reverse=True)]
    cdef int num_workers = cpu_count()

    cdef BeamSearch beam_search = BeamSearch(
        beam_candidate(0.0, greedy_correct_c / total_c, greedy_correct_c / total_c, False, [], []),
        dest_vec,
        tags_vec,
        times_pointed_vec,
        <long>cached_satisfy.ctypes.data,
        rows,
        cols,
        all_ids,
        all_aucs,
        num_workers
    )

    cdef int remaining = beam_search.candidates.size()
    cdef int work_size = all_ids.size()
    cdef int worker_idx = 0
    cdef int iteration = 0
    cdef int i = 0
    while True:
        threads = []
        total_work = np.zeros(num_workers, dtype=np.int32)
        pbar = get_progress_bar("disambiguate", max_value=beam_search.candidates.size() * work_size,
                                item="relations")


        for worker_idx in range(num_workers):
            threads.append(
                Thread(target=beam_project_worker,
                       args=(worker_idx,
                             num_workers,
                             penalty,
                             greedy_correct_c,
                             total_c,
                             work_size,
                             beam_width,
                             total_work,
                             iteration,
                             subset,
                             beam_search,
                             pbar),
                       daemon=True)
            )
        pbar.start()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        with nogil:
            # concatenate all beam outputs:
            for worker_idx in range(1, num_workers):
                beam_search.new_candidates[0].insert(
                    beam_search.new_candidates[0].end(),
                    beam_search.new_candidates[worker_idx].begin(),
                    beam_search.new_candidates[worker_idx].end()
                )
            stdsort(
                beam_search.new_candidates[0].begin(),
                beam_search.new_candidates[0].end(),
                &candidate_order)
            if subset > 0 and iteration == 0:
                for i in range(beam_search.new_candidates[0].size()):
                    if beam_search.new_candidates[0][i].kept_indices.size() == 1:
                        beam_search.subset_indices.push_back(
                            beam_search.new_candidates[0][i].kept_indices[0]
                        )
                        beam_search.subset_aucs.push_back(
                            beam_search.new_candidates[0][i].sum_auc
                        )
                if beam_search.subset_indices.size() > subset:
                    beam_search.subset_indices.erase(
                        beam_search.subset_indices.begin() + subset,
                        beam_search.subset_indices.end())
                    beam_search.subset_aucs.erase(
                        beam_search.subset_aucs.begin() + subset,
                        beam_search.subset_aucs.end())
                work_size = beam_search.subset_indices.size()

            beam_search.new_candidates[0].erase(
                beam_search.new_candidates[0].begin() + beam_width,
                beam_search.new_candidates[0].end())
            beam_search.candidates = beam_search.new_candidates[0]

        pbar.finish()
        report = "score: %.5f [objective: %.5f (%d items)]" % (
            beam_search.candidates[0].secondary_objective,
            beam_search.candidates[0].objective,
            beam_search.candidates[0].kept_indices.size()
        )
        print(report)
        if log is not None:
            with open(log, "at") as fout:
                fout.write(report + "\n")
        with nogil:
            remaining = beam_search.candidates.size()
            for beam_idx in range(beam_search.candidates.size()):
                if beam_search.candidates[beam_idx].stopped:
                    remaining -= 1
            if remaining == 0:
                break
        iteration += 1

    kept = []
    all_qids_relations = list(sorted(aucs.keys()))
    for k, idx in enumerate(beam_search.candidates[0].kept_indices):
        qid, relation = all_qids_relations[idx]
        kept.append({"qid": qid, "relation": relation})
    return kept, beam_search.candidates[0].secondary_objective


@cython.boundscheck(False)
@cython.wraparound(False)
def cem_project(cached_satisfy, key2row, tags, aucs, ids,
                float penalty, float best_frac=0.2, int n_samples=1000,
                int n_itr=500, log=None):
    all_classifications = None
    greedy_correct, total, remainder_tags = greedy_disambiguate(tags)
    cdef int greedy_correct_c = greedy_correct
    cdef double total_c = total
    cdef int current_best = greedy_correct
    cdef vector[vector[int]] tags_vec
    cdef vector[vector[int]] times_pointed_vec
    cdef vector[int] dest_vec
    cdef long rows = cached_satisfy.shape[0]
    cdef long cols = cached_satisfy.shape[1]
    for dest, other_dest, times_pointed in remainder_tags:
        tags_vec.push_back(other_dest)
        dest_vec.push_back(dest)
        times_pointed_vec.push_back(times_pointed)
    cdef bool* all_classifications_ptr = bool_ptr(cached_satisfy)

    cdef vector[int] proposal
    cdef unordered_set[int] proposal_set
    cdef vector[int] all_ids = [key2row[(qid, relation)] for (qid, relation) in sorted(aucs.keys(), key=lambda x: aucs[x], reverse=True)]
    cdef vector[float] all_aucs = [aucs[key] for key in sorted(aucs.keys(), key=lambda x: aucs[x], reverse=True)]

    cdef double improvement = 0
    cdef int parameter_size = len(aucs)
    cdef np.ndarray[float, ndim=1] cur_mean = np.zeros(parameter_size, dtype=np.float32)
    cur_mean += min(0.5, 50.0 / (<double>parameter_size))
    cdef np.ndarray[float, ndim=2] xs
    cdef int sample_idx = 0
    cdef vector[BeamCandidate] candidates
    cdef double sum_auc = 0.0
    cdef BeamCandidate best_x
    cdef int v = 0
    cdef bool all_binary = False
    cdef int to_keep = max(1, <int>((<float>n_samples) * best_frac))

    with nogil:
        for itr in range(n_itr):
            with gil:
                pbar = get_progress_bar("Sampling", max_value=n_samples, item='samples')
                pbar.start()
                xs = <np.ndarray[float, ndim=2]>(np.random.binomial(
                    1, cur_mean, size=(n_samples, parameter_size)
                ).astype(np.float32))
            candidates.clear()

            for sample_idx in range(n_samples):
                proposal.clear()
                proposal_set.clear()
                sum_auc = 0.0
                for v in range(parameter_size):
                    if xs[sample_idx, v] > 0:
                        proposal.push_back(v)
                        proposal_set.insert(v)
                        sum_auc += all_aucs[v]
                improvement = fast_disambiguate(
                    dest_vec,
                    tags_vec,
                    times_pointed_vec,
                    all_classifications_ptr,
                    rows,
                    cols,
                    proposal
                )
                if proposal.size() == 0:
                    objective = greedy_correct_c
                else:
                    objective = (
                        (greedy_correct_c + improvement * (sum_auc / proposal.size())) / total_c -
                        # number of items is penalized
                        proposal.size() * penalty
                    )
                candidates.push_back(beam_candidate(
                    sum_auc,
                    (greedy_correct_c + improvement) / total_c,
                    objective,
                    False,
                    proposal,
                    proposal_set
                ))
                if sample_idx % 50 == 0:
                    with gil:
                        pbar.update(sample_idx)
            stdsort(candidates.begin(), candidates.end(), &candidate_order)
            candidates.erase(candidates.begin() + to_keep, candidates.end())
            for v in range(parameter_size):
                cur_mean[v] = 0
            for sample_idx in range(candidates.size()):
                for v in candidates[sample_idx].kept_indices:
                    cur_mean[v] += 1.0 / candidates.size()
            for v in range(parameter_size):
                if cur_mean[v] > 1:
                    cur_mean[v] = 1.0

            best_x = candidates[0]
            with gil:
                pbar.finish()
                report = "%d/%d: score: %.5f [objective: %.5f (%d items)]" % (
                    itr,
                    n_itr,
                    best_x.secondary_objective,
                    best_x.objective,
                    best_x.kept_indices.size()
                )
                print(report)
                if log is not None:
                    with open(log, "at") as fout:
                        fout.write(report + "\n")

            all_binary = True
            for v in range(parameter_size):
                if cur_mean[v] > 1e-6 and cur_mean[v] < (1.0 - 1e-6):
                    all_binary = False
                    break
            if all_binary:
                with gil:
                    print("mean is binary -- stopping early")
                break
    kept = []
    all_qids_relations = list(sorted(aucs.keys(), key=lambda x: aucs[x], reverse=True))
    for k, idx in enumerate(best_x.kept_indices):
        qid, relation = all_qids_relations[idx]
        kept.append({"qid": qid, "relation": relation})
    return kept, best_x.secondary_objective


cdef void evaluate_pop(const vector[int]& dest_vec,
                       const vector[vector[int]]& tags_vec,
                       const vector[vector[int]]& times_pointed_vec,
                       bool* all_classifications_ptr,
                       long rows,
                       long cols,
                       vector[float]& all_aucs,
                       int greedy_correct_c,
                       double total_c,
                       float penalty,
                       samples):
    cdef vector[vector[int]] all_kept_indices
    cdef vector[double] all_sum_auc
    cdef vector[double] all_objective
    cdef vector[double] all_secondary
    cdef double sum_auc = 0.0
    cdef double improvement = 0
    cdef double objective = 0
    cdef int v = 0
    cdef int sample_idx = 0

    for sample in samples:
        all_kept_indices.push_back(np.where(sample)[0])
        with nogil:
            sum_auc = 0.0
            for v in all_kept_indices[all_kept_indices.size() - 1]:
                sum_auc += all_aucs[v]
            all_sum_auc.push_back(sum_auc)

    if len(samples) > 99:
        pbar = get_progress_bar("Sampling", max_value=len(samples), item='samples')
        pbar.start()
    else:
        pbar = None

    with nogil:
        for sample_idx in range(all_kept_indices.size()):
            improvement = fast_disambiguate(
                dest_vec,
                tags_vec,
                times_pointed_vec,
                all_classifications_ptr,
                rows,
                cols,
                all_kept_indices[sample_idx]
            )
            objective = (
                (greedy_correct_c + improvement * (all_sum_auc[sample_idx] / all_kept_indices[sample_idx].size())) / total_c -
                # number of items is penalized
                all_kept_indices[sample_idx].size() * penalty
            )
            all_objective.push_back(objective)
            all_secondary.push_back((greedy_correct_c + improvement) / total_c)
            if sample_idx % 50 == 0:
                with gil:
                    if pbar is not None:
                        pbar.update(sample_idx)
    if pbar is not None:
        pbar.finish()
    for sample_idx in range(all_kept_indices.size()):
        samples[sample_idx].fitness.values = (all_objective[sample_idx],)
        samples[sample_idx].secondary = all_secondary[sample_idx]



def ga_project(cached_satisfy, key2row, tags, aucs, ids, penalty,
               ngen=40, n_samples=1000, cxpb=0.2, mutpb=0.5,
               log=None):
    all_classifications = None
    greedy_correct, total, remainder_tags = greedy_disambiguate(tags)
    cdef int greedy_correct_c = greedy_correct
    cdef double total_c = total
    cdef int current_best = greedy_correct
    cdef vector[vector[int]] tags_vec
    cdef vector[vector[int]] times_pointed_vec
    cdef vector[int] dest_vec
    cdef long rows = cached_satisfy.shape[0]
    cdef long cols = cached_satisfy.shape[1]
    for dest, other_dest, times_pointed in remainder_tags:
        tags_vec.push_back(other_dest)
        dest_vec.push_back(dest)
        times_pointed_vec.push_back(times_pointed)
    cdef bool* all_classifications_ptr = bool_ptr(cached_satisfy)

    cdef vector[int] proposal
    cdef vector[float] all_aucs = [aucs[key] for key in sorted(aucs.keys())]

    deap_creator.create("FitnessMax", deap_base.Fitness, weights=(1.0,))
    deap_creator.create("Individual", np.ndarray, fitness=deap_creator.FitnessMax, secondary=float)
    toolbox = deap_base.Toolbox()
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", deap_tools.mutFlipBit, indpb=10.0 / len(aucs))
    toolbox.register("select", deap_tools.selTournament, tournsize=3)
    scale = 50.0/len(aucs)
    pop = [deap_creator.Individual(np.random.binomial(1, p=scale, size=len(aucs)).astype(np.bool))
           for i in range(n_samples)]

    evaluate_pop(
        dest_vec,
        tags_vec,
        times_pointed_vec,
        all_classifications_ptr,
        rows,
        cols,
        all_aucs,
        greedy_correct_c,
        total_c,
        penalty,
        pop
    )

    for g in range(ngen):
        t0 = time.time()
        pop = toolbox.select(pop, k=len(pop))
        pop = deap_algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        invalids = [ind for ind in pop if not ind.fitness.valid]
        evaluate_pop(
            dest_vec,
            tags_vec,
            times_pointed_vec,
            all_classifications_ptr,
            rows,
            cols,
            all_aucs,
            greedy_correct_c,
            total_c,
            penalty,
            invalids
        )
        best_x = max(pop, key=lambda x: x.fitness.values[0])
        t1 = time.time()
        report = "%d/%d: score: %.5f [objective: %.5f (%d items)], time %.3fs" % (
            g,
            ngen,
            float(best_x.secondary),
            float(best_x.fitness.values[0]),
            best_x.sum(),
            t1 - t0
        )
        print(report)
        if log is not None:
            with open(log, "at") as fout:
                fout.write(report + "\n")
    return [{"qid": qid, "relation": relation_name}
            for keep, (qid, relation_name)
            in zip(best_x, sorted(aucs.keys())) if keep], float(best_x.secondary)

