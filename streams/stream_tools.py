from itertools import islice, tee


def take(n, seq):
    """
    >>> list(take(2, 'abcd'))
    ['a', 'b']
    """
    return islice(seq, n)


def drop(n, seq):
    """
    >>> list(drop(2, 'abcd'))
    ['c', 'd']
    """
    return islice(seq, n, None)


def successive_pairs_stream(stream):
    """
    Return successive pairs from the (potentially infinite) input stream

    >>> list(successive_pairs_stream('abcd'))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    """
    s1, s2 = tee(stream)
    next(s2)
    return zip(s1, s2)
