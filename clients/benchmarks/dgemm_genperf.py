#!/usr/bin/python

from yaml import dump
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

import sys
import argparse

parser = argparse.ArgumentParser(description="Generate YAML file for DGEMM tests")
parser.add_argument("-o", "--out", dest='outfile', type=argparse.FileType('w'),
                    default=sys.stdout)
parsed = parser.parse_args()

PerfTests = []
hit = set()


def add_test(dict):
    sig = ':'.join([str(dict[k]) for k in 'transA', 'transB', 'M', 'N', 'K',
                    'lda', 'ldb', 'ldc', 'alpha', 'beta'])
    if sig not in hit:
        hit.add(sig)
        PerfTests.append(dict)


for N in (45000, 63000):
    for NB in xrange(256, 1025, 128):
        k = NB
        for n in (1 if t == 0 else t for t in xrange(0, N+1, NB)):
            for m in (n-1, n):
                if m > 0:
                    add_test({'transA': 'N',
                              'transB': 'N',
                              'M': m,
                              'N': n,
                              'K': k,
                              'lda': N,
                              'ldb': N,
                              'ldc': N,
                              'alpha': 1.0,
                              'beta': 0.0,
                              'name': 'GEMM Perf',
                              'category': 'stage1.1',
                              'function': 'testing_gemm',
                              'type': 'double',
                              })

        for x in xrange(1, 11):
            for n in (x*k, x*k+1):
                for m in (1 if t == 0 else t for t in xrange(0, N+1, NB)):
                    add_test({'transA': 'N',
                              'transB': 'T',
                              'M': m,
                              'N': n,
                              'K': k,
                              'lda': N,
                              'ldb': N,
                              'ldc': N,
                              'alpha': 1.0,
                              'beta': 0.0,
                              'name': 'GEMM Perf',
                              'category': 'stage1.2',
                              'function': 'testing_gemm',
                              'type': 'double',
                              })

        for m in (1 if t == 0 else t for t in xrange(0, N+1, NB)):
            for n in xrange(1, NB):
                k = n
                add_test({'transA': 'N',
                          'transB': 'N',
                          'M': m,
                          'N': n,
                          'K': k,
                          'lda': N,
                          'ldb': N,
                          'ldc': N,
                          'alpha': 1.0,
                          'beta': 0.0,
                          'name': 'GEMM Perf',
                          'category': 'stage2',
                          'function': 'testing_gemm',
                          'type': 'double',
                          })


dump({'Tests': PerfTests}, parsed.outfile, Dumper=Dumper)
