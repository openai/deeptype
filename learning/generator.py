import queue
import threading


def prefetch_generator(generator, to_fetch=10):
    q = queue.Queue(maxsize=to_fetch)

    def thread_worker(queue, gen):
        for val in gen:
            queue.put(val)
        queue.put(None)

    t = threading.Thread(target=thread_worker, args=(q, generator))
    some_exception = None
    try:
        t.start()
        while True:
            job = q.get()
            if job is None:
                break
            yield job
            del job
            # print("q.qsize() %d" % (q.qsize(),), flush=True)
    except Exception as e:
        some_exception = e
    finally:
        if some_exception is not None:
            raise some_exception
        t.join()
    del t
