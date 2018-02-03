cimport cython

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
import marisa_trie

from .progressbar import get_progress_bar
from .bash import count_lines
from .anchor_filtering import clean_up_trie_source
from multiprocessing import cpu_count, Queue
from threading import Thread

from libc.stdio cimport sscanf, FILE
from libc.string cimport strchr

cdef extern from "stdio.h" nogil:
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *)
    #int fclose ( FILE * stream )
    int fclose(FILE *)
    #ssize_t getline(char **lineptr, size_t *n, FILE *stream);
    ssize_t getline(char **, size_t *, FILE *)


cdef class RedirectionsHolder(object):
    cdef unordered_map[string, string] _redirections

    def __init__(self, path):
        filename_byte_string = path.encode("utf-8")
        cdef char* fname = filename_byte_string
        cdef FILE* cfile
        cfile = fopen(fname, "rb")
        if cfile == NULL:
            raise FileNotFoundError(2, "No such file: '%s'" % (path,))
        cdef char *line = NULL
        cdef size_t l = 0
        cdef size_t read
        cdef char[256] source
        cdef char[256] dest
        cdef char* uppercased_in_python
        cdef char* tab_pos
        cdef char* end_pos

        with nogil:
            while True:
                read = getline(&line, &l, cfile)
                if read == -1:
                    break

                tab_pos = strchr(line, '\t')
                if (tab_pos - line) > 256 or tab_pos == NULL:
                    continue
                end_pos = strchr(tab_pos, '\n')
                if (end_pos - tab_pos) > 256:
                    continue
                return_code = sscanf(line, "%256[^\n\t]\t%256[^\n]", &source, &dest)
                if return_code != 2:
                    continue
                with gil:
                    decoded = source.decode("utf-8")
                    decoded = (decoded[0].upper() + decoded[1:]).encode("utf-8")
                    uppercased_in_python = decoded
                self._redirections[string(uppercased_in_python)] = string(dest)
        fclose(cfile)

    def __len__(self):
        return self._redirections.size()

    def __contains__(self, key):
        return self._redirections.find(key.encode("utf-8")) != self._redirections.end()

    def __getitem__(self, key):
        cdef unordered_map[string, string].iterator finder = self._redirections.find(key.encode("utf-8"))
        if finder == self._redirections.end():
            raise KeyError(key)
        return deref(finder).second.decode("utf-8")

    def get(self, key, default=None):
        cdef unordered_map[string, string].iterator finder = self._redirections.find(key.encode("utf-8"))
        if finder == self._redirections.end():
            return default
        return deref(finder).second.decode("utf-8")

    def _asdict(self):
        out = {}
        for kv in self._redirections:
            out[kv.first.decode("utf-8")] = kv.second.decode("utf-8")
        return out


def load_redirections(path):
    return RedirectionsHolder(path)


@cython.boundscheck(False)
@cython.wraparound(False)
def successor_mask(np.ndarray[int, ndim=1] values,
                   np.ndarray[int, ndim=1] offsets,
                   bad_node_pair_right,
                   np.ndarray[int, ndim=1] active_nodes):
    np_dest_array = np.zeros(len(offsets), dtype=np.bool)
    cdef bool* dest_array = bool_ptr(np_dest_array)

    cdef unordered_map[int, vector[int]] bad_node_pair_right_c
    cdef unordered_map[int, vector[bool]] bad_node_pair_right_c_backups
    for item, value in bad_node_pair_right.items():
        bad_node_pair_right_c[item] = value
        bad_node_pair_right_c_backups[item] = vector[bool](len(value), 0)
    cdef int i = 0
    cdef int j = 0
    cdef int active_nodes_max = active_nodes.shape[0]
    cdef int end = 0
    cdef int start = 0
    cdef int active_node = 0
    cdef int [:] subvalues = values
    cdef int [:] values_view = values
    cdef int* bad_node_pair_right_c_ptr
    cdef vector[bool]* bad_node_pair_right_c_backups_ptr

    with nogil:
        for i in range(active_nodes_max):
            active_node = active_nodes[i]
            end = offsets[active_node]
            if active_node == 0:
                start = 0
            else:
                start = offsets[active_node - 1]
            if bad_node_pair_right_c.find(active_node) != bad_node_pair_right_c.end():
                bad_node_pair_right_c_ptr = bad_node_pair_right_c[active_node].data()
                bad_node_pair_right_c_backups_ptr = &bad_node_pair_right_c_backups[active_node]
                for j in range(bad_node_pair_right_c[active_node].size()):
                    bad_node_pair_right_c_backups_ptr[0][j] = dest_array[bad_node_pair_right_c_ptr[j]]

                subvalues = values_view[start:end]
                for j in range(end - start):
                    dest_array[subvalues[j]] = 1
                for j in range(bad_node_pair_right_c[active_node].size()):
                    dest_array[bad_node_pair_right_c_ptr[j]] = bad_node_pair_right_c_backups_ptr[0][j]
            else:
                subvalues = values_view[start:end]
                for j in range(end - start):
                    dest_array[subvalues[j]] = 1
    return np_dest_array

@cython.boundscheck(False)
@cython.wraparound(False)
def invert_relation(np.ndarray[int, ndim=1] values,
                    np.ndarray[int, ndim=1] offsets):

    cdef np.ndarray[int, ndim=1] new_values = np.empty_like(values)
    cdef np.ndarray[int, ndim=1] new_offsets = np.empty_like(offsets)

    cdef int max_offsets = len(offsets)
    cdef vector[vector[int]] inverted_edges = vector[vector[int]](max_offsets)
    cdef int so_far = 0
    cdef int i = 0
    cdef int j = 0
    cdef int[:] new_values_view = new_values
    cdef int[:] new_offsets_view = new_offsets
    cdef int[:] vector_view
    cdef int position = 0
    with nogil:
        for i in range(max_offsets):
            for j in range(offsets[i] - so_far):
                inverted_edges[values[so_far + j]].push_back(i)
            so_far = offsets[i]
        for i in range(max_offsets):
            for j in range(inverted_edges[i].size()):
                new_values_view[position + j] = inverted_edges[i][j]
            position += inverted_edges[i].size()
            new_offsets_view[i] = position
    return new_values, new_offsets


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_to_offset_array(list_list):
    cdef int num_values = len(list_list)
    cdef np.ndarray[int, ndim=1] offsets = np.zeros(num_values, dtype=np.int32)
    cdef int total_num_values = sum(len(v) for v in list_list)
    cdef np.ndarray[int, ndim=1] values = np.zeros(total_num_values, dtype=np.int32)
    cdef int[:] values_view = values
    cdef vector[int] list_list_i
    cdef int position = 0
    cdef int i = 0
    cdef int j = 0

    with nogil:
        for i in range(num_values):
            with gil:
                list_list_i = list_list[i]
            for j in range(list_list_i.size()):
                values_view[position + j] = list_list_i[j]
            position += list_list_i.size()
            offsets[i] = position
    return values, offsets


@cython.boundscheck(False)
@cython.wraparound(False)
def make_sparse(np.ndarray[int, ndim=1] dense):
    cdef np.ndarray[int, ndim=1] deltas = np.zeros_like(dense)
    deltas[1:] = dense[1:] - dense[:dense.shape[0] - 1]
    deltas[0] = dense[0]
    cdef np.ndarray[int, ndim=1] indices = np.nonzero(deltas)[0].astype(np.int32)
    cdef int original_length = len(deltas)
    cdef int num_nonzero = len(indices)
    cdef int i = 0
    cdef np.ndarray[int, ndim=1] out = np.zeros(num_nonzero * 2 + 1, dtype=np.int32)
    with nogil:
        # keep length around:
        out[0] = original_length
        for i in range(num_nonzero):
            # place index:
            out[i * 2 + 1] = indices[i]
            # place value:
            out[i * 2 + 2] = deltas[indices[i]]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def make_dense(np.ndarray[int, ndim=1] array, cumsum=False):
    cdef np.ndarray[int, ndim=1] out = np.zeros(array[0], dtype=np.int32)
    cdef int total_size = len(array)
    cdef int i = 0
    with nogil:
        for i in range(1, total_size, 2):
            out[array[i]] = array[i + 1]
    if cumsum:
        np.cumsum(out, out=out)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def offset_values_mask(np.ndarray[int, ndim=1] values,
                       np.ndarray[int, ndim=1] offsets,
                       np.ndarray[int, ndim=1] active_nodes):
    np_dest_array = np.zeros(len(values), dtype=np.bool)
    cdef bool* dest_array = bool_ptr(np_dest_array)
    cdef int i = 0
    cdef int j = 0
    cdef int active_nodes_max = len(active_nodes)
    cdef int end = 0
    cdef int start = 0
    cdef int active_node = 0
    with nogil:
        for i in range(active_nodes_max):
            active_node = active_nodes[i]
            end = offsets[active_node]
            if active_node == 0:
                start = 0
            else:
                start = offsets[active_node - 1]
            for j in range(end - start):
                dest_array[start + j] = 1
    return np_dest_array


@cython.boundscheck(False)
@cython.wraparound(False)
def remove_offset_array_negatives(np.ndarray[int, ndim=1] values,
                                  np.ndarray[int, ndim=1] offsets):
    cdef int position = 0
    cdef np.ndarray[int, ndim=1] values_out = np.zeros_like(values)
    cdef np.ndarray[int, ndim=1] offsets_out = np.zeros_like(offsets)
    cdef int start = 0
    cdef int end = 0
    cdef int i = 0
    cdef int j = 0
    cdef int max_offsets = len(offsets)
    with nogil:
        for i in range(max_offsets):
            end = offsets[i]
            for j in range(start, end):
                if values[j] > -1:
                    values_out[position] = values[j]
                    position += 1
            offsets_out[i] = position
            start = end
    return values_out[:position], offsets_out


@cython.boundscheck(False)
@cython.wraparound(False)
def related_promote_highest(np.ndarray[int, ndim=1] values,
                            np.ndarray[int, ndim=1] offsets,
                            np.ndarray[int, ndim=1] counts,
                            condition,
                            alternative,
                            int keep_min=5):
    cdef bool* condition_ptr = bool_ptr(condition)
    cdef bool* alternative_ptr = bool_ptr(alternative)
    cdef np.ndarray[int, ndim=1] new_values = values.copy()
    cdef int i = 0
    cdef int j = 0
    cdef int start = 0
    cdef int end = 0
    cdef int [:] new_values_view = new_values
    cdef int [:] values_view = values
    cdef int [:] counts_view = counts
    cdef int [:] offsets_view = offsets
    cdef int max_offsets = len(offsets)
    cdef int alternate_count = -1
    cdef bint any_switchers
    cdef int alternate_value = -1
    with nogil:
        for i in range(max_offsets):
            end = offsets_view[i]
            any_switchers = False
            alternate_value = -1
            alternate_count = -1
            for j in range(start, end):
                if condition_ptr[j] and values_view[j] > -1:
                    any_switchers = True
                if alternative_ptr[j]:
                    if counts_view[j] > alternate_count and values_view[j] > -1:
                        alternate_count = counts_view[j]
                        alternate_value = values_view[j]
            if any_switchers and alternate_value > -1:
                if alternate_count <= keep_min:
                    alternate_value = -1
                for j in range(start, end):
                    if condition_ptr[j] and values_view[j] > -1:
                        if counts_view[j] < alternate_count:
                            new_values[j] = alternate_value
            start = end
    return new_values

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void single_usage_extend_cursor(unordered_set[int]& destination_cursor,
                                     unordered_set[int] cursor,
                                     int* offset,
                                     int* values) nogil:
    destination_cursor.clear()
    cdef int start = 0
    cdef int i = 0
    cdef int val
    for val in cursor:
        start = 0 if val == 0 else offset[val - 1]
        for i in range(start, offset[val]):
            if values[i] > -1:
                destination_cursor.insert(values[i])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void extend_cursor(unordered_set[int]& destination_cursor,
                        unordered_set[int] newcomers,
                        int* offset,
                        int* values,
                        int usage) nogil:
    destination_cursor.clear()
    cdef int start = 0
    cdef int i = 0
    cdef int val
    cdef int destination_cursor_size = destination_cursor.size()
    cdef int uses = 0
    cdef unordered_set[int] new_newcomers
    while uses < usage:
        new_newcomers.clear()
        for val in newcomers:
            start = 0 if val == 0 else offset[val - 1]
            for i in range(start, offset[val]):
                # totally new item being explored:
                if values[i] > -1:
                    if destination_cursor.find(values[i]) == destination_cursor.end():
                        new_newcomers.insert(values[i])
                        destination_cursor.insert(values[i])
        uses += 1
        if new_newcomers.size() == 0:
            break
        else:
            newcomers = new_newcomers


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint keep_high_or_highest(int original_count, int new_best_count) nogil:
    return (new_best_count > original_count) or (new_best_count > 50 and new_best_count >= 0.8 * original_count)

ctypedef int* int_ptr


def binary_search(v_min, v_max, test):
    l, r, mid, solution = v_min, v_max, -1, -1
    while l <= r:
        mid = (l + r) // 2
        if test(mid):
            solution = mid
            r = mid - 1
        else:
            l = mid + 1
    return solution


def allocate_work(arr, max_work):
    last = 0
    sol = []
    work_so_far = 0

    while last < len(arr):
        next_pt = binary_search(
            last,
            len(arr) - 1,
            lambda point: arr[point] > work_so_far + max_work
        )
        if next_pt == -1:
            next_pt = len(arr)
        work_so_far = arr[next_pt - 1]
        if last == next_pt:
            return None
        sol.append((last, next_pt))
        last = next_pt
    return sol


def fair_work_allocation(offsets, num_workers):
    def check_size(work_size):
        allocated = allocate_work(offsets, work_size)
        return allocated is not None and len(allocated) <= num_workers

    best_work_size = binary_search(
        np.ceil(offsets[-1] / num_workers),
        offsets[-1],
        check_size
    )
    return allocate_work(offsets, best_work_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void extend_relations_find_largest_parent(int j, int start, int end,
                                               const unordered_map[int, bint]& is_parent,
                                               int current_count,
                                               int keep_min,
                                               int [:] counts_view,
                                               int [:] values_view,
                                               int [:] new_values_view,
                                               bool* alternative) nogil:
    cdef int max_count = -1
    cdef int max_value = -1
    cdef int oj
    if is_parent.size() > 0:
        for oj in range(start, end):
            # if the new entity is the parent according to some
            # hard rule, or the number of links to the parent
            # is greater than those to the child, consider swapping:
            if (alternative[oj] and values_view[oj] > -1 and
                is_parent.find(values_view[oj]) != is_parent.end() and
                (
                    is_parent.at(values_view[oj]) or
                    keep_high_or_highest(current_count, counts_view[oj])
                ) and counts_view[oj] > max_count) and values_view[oj] != values_view[j]:

                max_count = counts_view[oj]
                max_value = values_view[oj]
        if max_value > -1:
            if max_count > keep_min:
                new_values_view[j] = max_value
            else:
                new_values_view[j] = -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void extend_relations_extend_path(int j,
                                       int path_idx,
                                       int [:] values_view,
                                       const vector[vector[int_ptr]]& relation_data_offsets,
                                       const vector[vector[int_ptr]]& relation_data_values,
                                       const vector[vector[int]]& relation_data_max_usage,
                                       bool promote,
                                       unordered_map[int, bint]& is_parent) nogil:
    cdef unordered_set[int] cursor
    # now starting at the current point j
    # try to walk towards the entity by extending using paths
    if values_view[j] > -1:
        cursor.insert(values_view[j])
    for step_idx in range(relation_data_offsets[path_idx].size()):
        # path can only be used once:
        if relation_data_max_usage[path_idx][step_idx] == 1:
            single_usage_extend_cursor(
                cursor,
                cursor,
                relation_data_offsets[path_idx][step_idx],
                relation_data_values[path_idx][step_idx]
            )
        else:
            # path can be used recursively:
            extend_cursor(
                cursor,
                cursor,
                relation_data_offsets[path_idx][step_idx],
                relation_data_values[path_idx][step_idx],
                relation_data_max_usage[path_idx][step_idx]
            )
        # if no entity was found in this process, stop
        if cursor.size() == 0:
            break
    # if there are entities connected via extending the cursor
    # then pick the largest one of those as the parent:
    if cursor.size() > 0:
        for val in cursor:
            is_parent_finder = is_parent.find(val)
            if is_parent_finder == is_parent.end():
                is_parent[val] = promote
            elif promote:
                is_parent[val] = True


cdef bool* bool_ptr(array):
    if array.dtype != np.bool:
        raise ValueError("Can only take boolean pointer from "
                         "array with dtype np.bool (got %r)" % (array.dtype))
    return <bool*>(<long>array.ctypes.data)

@cython.boundscheck(False)
@cython.wraparound(False)
def extend_relations_worker(int worker_idx,
                            relation_data,
                            job_queue,
                            np.ndarray[int, ndim=1] new_values,
                            np.ndarray[int, ndim=1] values,
                            np.ndarray[int, ndim=1] offsets,
                            np.ndarray[int, ndim=1] counts,
                            np_alternative,
                            np.ndarray[int, ndim=1] total_work,
                            int keep_min,
                            pbar):
    cdef int [:] offsets_view = offsets
    cdef int [:] counts_view = counts
    cdef int [:] values_view = values
    cdef int [:] new_values_view = new_values
    cdef int [:] total_work_view = total_work

    cdef vector[vector[int_ptr]] relation_data_offsets
    cdef vector[vector[int_ptr]] relation_data_values
    cdef vector[vector[int]] relation_data_max_usage
    cdef vector[bool*] relation_data_condition
    cdef vector[bool] relation_data_promote

    cdef vector[int_ptr] step_offsets_single
    cdef vector[int_ptr] step_values_single
    cdef vector[int] step_max_usage_single
    cdef bool* alternative = bool_ptr(np_alternative)

    for path, path_condition, promote in relation_data:
        for step_offsets, step_values, max_usage in path:
            step_offsets_single.push_back(<int*> (<np.ndarray[int, ndim=1]>step_offsets).data)
            step_values_single.push_back(<int*> (<np.ndarray[int, ndim=1]>step_values).data)
            step_max_usage_single.push_back(max_usage)
        relation_data_offsets.push_back(step_offsets_single)
        relation_data_values.push_back(step_values_single)
        relation_data_max_usage.push_back(step_max_usage_single)
        relation_data_condition.push_back(bool_ptr(path_condition))
        relation_data_promote.push_back(promote)
        # clear temps just being used to load vectors
        step_offsets_single.clear()
        step_values_single.clear()
        step_max_usage_single.clear()

    cdef int max_offsets = len(offsets)
    cdef int num_paths = len(relation_data)

    cdef int i
    cdef int j
    cdef int oj
    cdef int start
    cdef int end
    cdef int path_idx
    cdef int current_count
    cdef vector[bint] all_paths_false

    for path_idx in range(num_paths):
        all_paths_false.push_back(False)

    cdef vector[bint] paths_active
    cdef unordered_map[int, bint] is_parent
    cdef unordered_map[int, bint].iterator is_parent_finder
    cdef bint any_is_alternative
    cdef int start_offset
    cdef int end_offset
    cdef int work_done = 0

    while True:
        job = job_queue.get()
        if job is None:
            break
        start_offset, end_offset = job
        start = offsets_view[start_offset - 1] if start_offset != 0 else 0
        with nogil:
            for i in range(start_offset, end_offset):
                end = offsets_view[i]
                if end - start > 1:
                    # check if any of the parents can be used
                    # as valid alternatives for the current
                    # item
                    any_is_alternative = False
                    for oj in range(start, end):
                        if alternative[oj]:
                            any_is_alternative = True
                            break
                    # if there is an alternative, look for a path
                    # that connects the current entity to this
                    # new alternative
                    if any_is_alternative:
                        paths_active = all_paths_false
                        for path_idx in range(num_paths):
                            for oj in range(start, end):
                                # look if a particular alternative can be used
                                # and if the specific entity that is being
                                # refered to is not -1 (e.g. masked out)
                                # and whether the path_idx path truth table
                                # is true at this location:
                                if (alternative[oj] and
                                    values_view[oj] > -1 and
                                    relation_data_condition[path_idx][values_view[oj]]):
                                    paths_active[path_idx] = True
                                    # mark that the path has at least one
                                    # possible entity connected to it and move on
                                    break
                        for j in range(start, end):
                            is_parent.clear()
                            current_count = counts_view[j]
                            for path_idx in range(num_paths):
                                # filter by paths that are connectible to the entity
                                # (see filtering above)
                                if paths_active[path_idx]:
                                    extend_relations_extend_path(
                                        j,
                                        path_idx,
                                        values_view,
                                        relation_data_offsets,
                                        relation_data_values,
                                        relation_data_max_usage,
                                        relation_data_promote[path_idx],
                                        is_parent)
                            # select largest new parent (from cursor)
                            # to replace the current entity
                            extend_relations_find_largest_parent(
                                j,
                                start,
                                end,
                                is_parent,
                                current_count,
                                keep_min,
                                counts_view,
                                values_view,
                                new_values_view,
                                alternative)
                start = end
                work_done += 1
                if work_done % 5000 == 0:
                    total_work_view[worker_idx] = work_done
                    if worker_idx == 0:
                        with gil:
                            pbar.update(total_work.sum())
    if worker_idx == 0:
        pbar.update(total_work.sum())

@cython.boundscheck(False)
@cython.wraparound(False)
def extend_relations(relation_data,
                     np.ndarray[int, ndim=1] values,
                     np.ndarray[int, ndim=1] offsets,
                     np.ndarray[int, ndim=1] counts,
                     alternative,
                     pbar,
                     keep_min=5,
                     job_factor=6):
    cdef np.ndarray[int, ndim=1] new_values = values.copy()

    # ensure bool is used:
    relation_data = [
        (path, path_condition, promote) if path_condition.dtype == np.bool else
        (path, path_condition.astype(np.bool), promote)
        for path, path_condition, promote in relation_data
    ]

    threads = []
    num_workers = cpu_count()
    cdef np.ndarray[int, ndim=1] total_work = np.zeros(
        num_workers, dtype=np.int32)
    allocations = fair_work_allocation(offsets, num_workers * job_factor)
    job_queue = Queue()
    for job in allocations:
        job_queue.put(job)
    for worker_idx in range(num_workers):
        job_queue.put(None)
    for worker_idx in range(num_workers):
        threads.append(
            Thread(target=extend_relations_worker,
                   args=(worker_idx,
                         relation_data,
                         job_queue,
                         new_values,
                         values,
                         offsets,
                         counts,
                         alternative,
                         total_work,
                         keep_min,
                         pbar),
                   daemon=True)
        )


    pbar.start()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    pbar.finish()
    return new_values

# TODO: create a special case for end - start == 3
@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_values(np.ndarray[int, ndim=1] offsets,
                  np.ndarray[int, ndim=1] values,
                  np.ndarray[int, ndim=1] counts):
    cdef int max_offsets = len(offsets)


    cdef np.ndarray[int, ndim=1] new_offsets = np.zeros_like(offsets)
    cdef np.ndarray[int, ndim=1] new_values = np.zeros_like(values)
    cdef np.ndarray[int, ndim=1] new_counts = np.zeros_like(counts)
    cdef np.ndarray[int, ndim=1] location_shift = np.zeros_like(values)

    cdef int [:] offsets_view = offsets
    cdef int [:] values_view = values
    cdef int [:] counts_view = counts

    cdef int [:] new_offsets_view = new_offsets
    cdef int [:] new_values_view = new_values
    cdef int [:] new_counts_view = new_counts

    cdef int i
    cdef int j
    cdef int start = 0
    cdef int end
    cdef unordered_map[int, int] obs
    cdef int pos = 0
    cdef int insertion_offset = 0
    cdef int index

    with nogil:
        for i in range(max_offsets):
            end = offsets_view[i]
            if end - start == 1:
                if values_view[start] > -1:
                    new_values_view[insertion_offset] = values_view[start]
                    new_counts_view[insertion_offset] = counts_view[start]
                    location_shift[start] = insertion_offset
                    new_offsets_view[i] = 1
                    insertion_offset += 1
                else:
                    new_offsets_view[i] = 0
            elif end - start == 2:
                if values_view[start] > -1 and values_view[start + 1] > -1:
                    if values_view[start] == values_view[start + 1]:
                        new_values_view[insertion_offset] = values_view[start]
                        new_counts_view[insertion_offset] = counts_view[start] + counts_view[start+1]
                        location_shift[start] = insertion_offset
                        location_shift[start + 1] = insertion_offset
                        insertion_offset += 1
                        new_offsets_view[i] = 1
                    else:
                        new_values_view[insertion_offset] = values_view[start]
                        new_counts_view[insertion_offset] = counts_view[start]
                        new_values_view[insertion_offset+1] = values_view[start+1]
                        new_counts_view[insertion_offset+1] = counts_view[start+1]
                        location_shift[start] = insertion_offset
                        location_shift[start + 1] = insertion_offset + 1
                        insertion_offset += 2
                        new_offsets_view[i] = 2
                elif values_view[start] > -1:
                    new_values_view[insertion_offset] = values_view[start]
                    new_counts_view[insertion_offset] = counts_view[start]
                    location_shift[start] = insertion_offset
                    location_shift[start + 1] = -1
                    insertion_offset += 1
                    new_offsets_view[i] = 1
                elif values_view[start + 1] > -1:
                    new_values_view[insertion_offset] = values_view[start+1]
                    new_counts_view[insertion_offset] = counts_view[start+1]
                    location_shift[start] = -1
                    location_shift[start + 1] = insertion_offset
                    insertion_offset += 1
                    new_offsets_view[i] = 1
                else:
                    new_offsets_view[i] = 0
            else:
                obs.clear()
                for j in range(start, end):
                    if values_view[j] > -1:
                        if obs.find(values_view[j]) == obs.end():
                            obs[values_view[j]] = insertion_offset
                            location_shift[j] = insertion_offset
                            new_values_view[insertion_offset] = values_view[j]
                            new_counts_view[insertion_offset] = counts_view[j]
                            insertion_offset += 1
                        else:
                            index = obs.at(values_view[j])
                            location_shift[j] = index
                            new_counts_view[index] += counts_view[j]
                    else:
                        location_shift[j] = -1
                new_offsets_view[i] = obs.size()
            start = end
    np.cumsum(new_offsets, out=new_offsets)
    new_values = new_values[:new_offsets[len(offsets) - 1]]
    new_counts = new_counts[:new_offsets[len(offsets) - 1]]
    return new_offsets, new_values, new_counts, location_shift


@cython.boundscheck(False)
@cython.wraparound(False)
def remap_offset_array(np.ndarray[int, ndim=1] mapping,
                       np.ndarray[int, ndim=1] offsets,
                       np.ndarray[int, ndim=1] values,
                       np.ndarray[int, ndim=1] counts):
    cdef int [:] mapping_view = mapping
    cdef int [:] offsets_view = offsets
    cdef int [:] values_view = values
    cdef int [:] counts_view = counts

    cdef np.ndarray[int, ndim=1] new_offsets = np.zeros_like(mapping)
    cdef np.ndarray[int, ndim=1] new_values = np.zeros_like(values)
    cdef np.ndarray[int, ndim=1] new_counts = np.zeros_like(counts)

    cdef int old_index,
    cdef int new_index
    cdef int start = 0
    cdef int j
    cdef int end
    cdef int old_start
    cdef int old_end
    cdef int max_offsets = len(mapping)

    with nogil:
        for new_index in range(max_offsets):
            old_index = mapping_view[new_index]
            if old_index > 0:
                old_start = offsets_view[old_index - 1]
            else:
                old_start = 0
            old_end = offsets_view[old_index]
            end = start + old_end - old_start
            for j in range(0, end - start):
                new_counts[start + j] = counts_view[old_start + j]
                new_values[start + j] = values_view[old_start + j]
            new_offsets[new_index] = end
            start = end
    return new_offsets, new_values, new_counts


@cython.boundscheck(False)
@cython.wraparound(False)
cdef build_trie_index2indices_array(vector[unordered_map[int, int]]& trie_index2indices):
    cdef np.ndarray[int, ndim=1] offsets = np.zeros(trie_index2indices.size(), dtype=np.int32)
    cdef int total_num_values = 0
    cdef int i
    with nogil:
        for i in range(trie_index2indices.size()):
            total_num_values += trie_index2indices[i].size()
    cdef np.ndarray[int, ndim=1] values = np.zeros(total_num_values, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] counts = np.zeros(total_num_values, dtype=np.int32)
    cdef int position = 0
    with nogil:
        for i in range(trie_index2indices.size()):
            for kv in trie_index2indices[i]:
                values[position] = kv.first
                counts[position] = kv.second
                position += 1
            offsets[i] = position
    return offsets, values, counts


def cleanup_title(dest):
    return (dest[0].upper() + dest[1:]).split('#')[0].replace('_', ' ')


def match_wikipedia_to_wikidata(dest, trie, redirections, prefix):
    prefixed_dest = prefix + "/" + dest
    dest_index = trie.get(prefixed_dest, -1)

    if dest_index == -1:
        cleaned_up_dest = cleanup_title(dest)
        prefixed_dest = prefix + "/" + cleaned_up_dest
        dest_index = trie.get(prefixed_dest, -1)

    if dest_index == -1:
        redirected_dest = redirections.get(cleaned_up_dest, None)
        if redirected_dest is not None:
            prefixed_dest = prefix + "/" + cleanup_title(redirected_dest)
            dest_index = trie.get(prefixed_dest, -1)
    if dest_index != -1:
        dest_index = dest_index[0][0]
    return dest_index


@cython.boundscheck(False)
@cython.wraparound(False)
def construct_mapping(anchor_trie,
                      anchor_tags,
                      wikipedia2wikidata_trie,
                      prefix,
                      redirections):
    filename_byte_string = anchor_tags.encode("utf-8")
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file: '%s'" % (anchor_tags,))

    cdef char *line = NULL
    cdef size_t l = 0
    cdef size_t read

    cdef vector[unordered_map[int, int]] trie_index2indices = vector[unordered_map[int, int]](len(anchor_trie))
    cdef vector[unordered_map[int, int]] trie_index2contexts = vector[unordered_map[int, int]](len(anchor_trie))

    cdef char[256] context
    cdef char[256] target
    cdef char[256] anchor
    cdef int anchor_int
    cdef int target_int
    cdef int context_int
    cdef int return_code
    cdef int num_lines = count_lines(anchor_tags)
    cdef int count = 0
    cdef char* tab_pos
    cdef char* end_pos
    pbar = get_progress_bar("Construct mapping", max_value=num_lines, item='lines')
    pbar.start()
    with nogil:
        while True:
            read = getline(&line, &l, cfile)
            if read == -1:
                break
            count += 1
            if count % 10000 == 0:
                with gil:
                    pbar.update(count)
            tab_pos = strchr(line, '\t')
            if (tab_pos - line) > 256 or tab_pos == NULL:
                continue
            end_pos = strchr(tab_pos, '\n')
            if (end_pos - tab_pos) > 256:
                continue
            return_code = sscanf(line, "%256[^\n\t]\t%256[^\n\t]\t%256[^\n\t]", &context, &anchor, &target)
            if return_code != 3:
                continue

            with gil:
                try:
                    target_int = match_wikipedia_to_wikidata(
                        target.decode("utf-8"),
                        wikipedia2wikidata_trie,
                        redirections,
                        prefix
                    )
                except UnicodeDecodeError:
                    continue

                if target_int != -1:
                    cleaned_up = clean_up_trie_source(anchor.decode("utf-8"))
                    if len(cleaned_up) > 0:
                        anchor_int = anchor_trie[cleaned_up]
                        context_int = match_wikipedia_to_wikidata(
                            context.decode("utf-8"),
                            wikipedia2wikidata_trie,
                            redirections,
                            prefix
                        )
                        with nogil:
                            trie_index2indices[anchor_int][target_int] += 1
                            trie_index2contexts[anchor_int][context_int] += 1
    fclose(cfile)
    pbar.finish()
    offsets, values, counts = build_trie_index2indices_array(trie_index2indices)
    context_offsets, context_values, context_counts = build_trie_index2indices_array(trie_index2contexts)
    return (
        (offsets, values, counts),
        (context_offsets, context_values, context_counts)
    )


def iterate_anchor_lines(anchor_tags,
                         redirections,
                         wikipedia2wikidata_trie,
                         prefix):
    filename_byte_string = anchor_tags.encode("utf-8")
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file: '%s'" % (anchor_tags,))

    cdef char *line = NULL
    cdef size_t l = 0
    cdef size_t read
    cdef char[256] context
    cdef char[256] target
    cdef char[256] anchor
    cdef string anchor_string
    cdef int anchor_int
    cdef int target_int
    cdef int context_int
    cdef int num_missing = 0
    cdef int num_broken = 0
    cdef int return_code
    cdef int num_lines = count_lines(anchor_tags)
    cdef int count = 0
    cdef char* tab_pos
    cdef char* end_pos
    cdef vector[pair[string, string]] missing
    cdef unordered_set[string] visited
    pbar = get_progress_bar("Construct mapping", max_value=num_lines, item='lines')
    pbar.start()
    with nogil:
        while True:
            read = getline(&line, &l, cfile)
            if read == -1:
                break
            count += 1
            if count % 1000 == 0:
                with gil:
                    pbar.update(count)
            tab_pos = strchr(line, '\t')
            if (tab_pos - line) > 256 or tab_pos == NULL:
                continue
            end_pos = strchr(tab_pos, '\n')
            if (end_pos - tab_pos) > 256:
                continue
            return_code = sscanf(line, "%256[^\n\t]\t%256[^\n\t]\t%256[^\n\t]", &context, &anchor, &target)
            if return_code != 3:
                num_broken += 1
                continue

            anchor_string = string(anchor)
            if visited.find(anchor_string) == visited.end():
                with gil:
                    try:
                        target_int = match_wikipedia_to_wikidata(
                            target.decode("utf-8"),
                            wikipedia2wikidata_trie,
                            redirections,
                            prefix
                        )
                    except UnicodeDecodeError:
                        num_broken += 1
                        continue

                    if target_int != -1:
                        with nogil:
                            visited.insert(anchor_string)
                        source = clean_up_trie_source(anchor.decode("utf-8"))
                        if len(source) > 0:
                            yield source
                    else:
                        num_missing += 1
                        with nogil:
                            missing.push_back(pair[string, string](anchor_string, string(target)))
    fclose(cfile)
    pbar.finish()
    print("%d/%d anchor_tags could not be found in wikidata" % (num_missing, num_lines))
    print("%d/%d anchor_tags links were malformed/too long" % (num_broken, num_lines))
    print("Missing anchor_tags sample:")
    cdef int i = 0
    for kv in missing:
        print("    " + kv.first.decode("utf-8") + " -> " + kv.second.decode("utf-8"))
        i += 1
        if i == 10:
            break

def construct_anchor_trie(anchor_tags, redirections, prefix, wikipedia2wikidata_trie):
    return marisa_trie.Trie(
        iterate_anchor_lines(
            anchor_tags=anchor_tags,
            wikipedia2wikidata_trie=wikipedia2wikidata_trie,
            redirections=redirections,
            prefix=prefix
        )
    )
