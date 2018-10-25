#!/usr/bin/python
"""Expand GEMM YAML test data file into binary Arguments records"""

import re
import sys
import ctypes

# Regex for type names in the YAML file. Optional *nnn indicates array.
TYPE_RE = re.compile(r'\w+(:?\s*\*\s*\d+)?$')

# Regex for integer ranges A..B[..C]
INT_RANGE_RE = re.compile(r'\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*(?:\.\.\s*(-?\d+)\s*)?$')

# Regex for Include dictionary entries
INCLUDE_RE = re.compile("(?i)include")

datatypes = {}
param = {}


def main():
    global datatypes, param

    # Parse YAML file
    (infile, param['outfile'], param['filter']) = parse_args()
    doc = get_doc(infile)

    # Return dictionary of all known datatypes
    datatypes = get_datatypes(doc)

    # Arguments structure corresponding to C/C++ structure
    param['Arguments'] = type('Arguments', (ctypes.Structure,),
                              {"_fields_": get_arguments(doc)})

    # Special names which get expanded as lists of arguments
    param['dict_lists_to_expand'] = doc.get('Dictionary lists to expand') or ()

    # Lists which are not expanded
    param['lists_to_not_expand'] = doc.get('Lists to not expand') or ()

    # Defaults
    defaults = doc.get('Defaults') or {}

    # Instantiate all of the tests, starting with defaults
    for test in doc['Tests']:
        case = defaults.copy()
        case.update(test)
        generate(case, instantiate)


def parse_args():
    '''Parse command-line arguments, returning input and output files'''

    import argparse
    parser = argparse.ArgumentParser(description="""
Expand GEMM YAML test data file into binary Arguments records
""")
    parser.add_argument('infile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('-o', '--out',
                        dest='outfile',
                        type=argparse.FileType('wb'),
                        default=sys.stdout)
    parser.add_argument('-f', '--filter',
                        dest='filter',
                        metavar="FILTER[,FILTER...]",
                        help="Filter tests based on nightly, pre_checkin, etc.",
                        type=lambda s: s.split(','))
    parsed = parser.parse_args()
    return (parsed.infile, parsed.outfile, parsed.filter)


def get_doc(infile):
    '''Parse the YAML file, handling !include files and include dictionaries'''
    try:          # Import either the C or pure-Python YAML parser
        from yaml import CLoader as BaseLoader
    except ImportError:
        from yaml import BaseLoader
    import yaml
    import os.path

    # Define a Loader class which handles !include directives in YAML
    class Loader(BaseLoader):
        def __init__(self, stream):
            self._root = os.path.split(stream.name)[0]
            super(Loader, self).__init__(stream)

        def include(self, node):
            filename = os.path.join(self._root, self.construct_scalar(node))
            with open(filename, 'r') as f:
                return yaml.load(f, Loader)

    Loader.add_constructor('!include', Loader.include)

    # Load the YAML document, recursively expanding !include directives
    doc = yaml.load(infile, Loader=Loader)

    # For every key beginning with "include", inject its contents into document
    for key in doc.keys():
        if INCLUDE_RE.match(key):
            inc = doc.pop(key)
            inc.pop('Definitions', None)   # Keep Definitons entries local
            doc.update(inc)

    return doc


def get_datatypes(doc):
    """ Get datatypes from YAML doc"""
    dt = ctypes.__dict__.copy()
    for declaration in doc.get('Datatypes') or ():
        for name, decl in declaration.items():
            if isinstance(decl, dict):
                # Create derived class type based on bases and attr entries
                dt[name] = type(name,
                                tuple([eval(t, dt)
                                       for t in decl.get('bases') or ()
                                       if TYPE_RE.match(t)]
                                      ), decl.get('attr') or {})
                # Import class' attributes into the datatype namespace
                for subtype in decl.get('attr') or {}:
                    if TYPE_RE.match(subtype):
                        dt[subtype] = eval(name+"."+subtype, dt)
            elif isinstance(decl, basestring) and TYPE_RE.match(decl):
                dt[name] = dt[decl]
            else:
                sys.exit("Unrecognized data type "+name+": "+repr(decl))
    return dt


def get_arguments(doc):
    """The kernel argument list, with argument names and types"""
    return [(var, eval(decl[var], datatypes))
            for decl in doc.get('Arguments') or ()
            if len(decl) == 1
            for var in decl
            if TYPE_RE.match(decl[var])]


def setdefaults(test):
    """Set default values for parameters"""
    # TODO: This should be ideally moved to YAML file, with eval'd expressions.
    if 'type' in test:
        test.setdefault('a_type', test['type'])
        test.setdefault('b_type', test['type'])
        test.setdefault('c_type', test['type'])
        test.setdefault('d_type', test['type'])
        test.setdefault('compute_type', test['type'])

    test.setdefault('lda',
                    test['M'] if test['transA'].upper() == 'N' else test['K'])
    test.setdefault('ldb',
                    test['K'] if test['transB'].upper() == 'N' else test['N'])
    test.setdefault('ldc', test['M'])
    test.setdefault('ldd', test['M'])

    test.setdefault('stride_a', test['lda'] *
                    (test['K'] if test['transA'].upper() == 'N' else test['M']))
    test.setdefault('stride_b', test['ldb'] *
                    (test['N'] if test['transB'].upper() == 'N' else test['K']))
    test.setdefault('stride_c', test['ldc'] * test['N'])
    test.setdefault('stride_d', test['ldd'] * test['N'])


def instantiate(test):
    """Instantiate a given test case"""

    # Filter based on test_class
    if param['filter'] and test.get("category") not in param['filter']:
        return

    test = test.copy()
    setdefaults(test)

    arguments = param['Arguments']._fields_

    # For type arguments, replace type name with type
    for typename in [decl[0] for decl in arguments
                     if decl[1] == datatypes.get('rocblas_datatype')]:
        test[typename] = datatypes[test[typename]]

    # For each argument declared in arguments, we generate a positional
    # argument in the Arguments constructor. For strings, we pass the
    # value of the string directly. For arrays, we unpack their contents
    # into the ctype array constructor and pass the ctype array. For
    # scalars, we coerce the string/numeric value into ctype.
    # This only works on Python 2.7+. It is incompatible with Python 3+.
    arg = param['Arguments'](*(
        test[name] if ctype._type_ == ctypes.c_char    # Strings
        else ctype(*test[name])                        # Arrays
        if issubclass(ctype, ctypes.Array)
        else ctype(test[name])                         # Scalars
        for (name, ctype) in arguments
    ))

    # Write the Arguments struct out to the binary file
    param['outfile'].write(bytearray(arg))


def generate(test, function):
    """Generate test combinations by iterating across lists recursively"""

    test = test.copy()

    # For specially named dictionary lists which list multiple argument sets,
    # they are expanded and merged into the original test argument list
    for key in param['dict_lists_to_expand']:
        if key in test:
            for item in test.pop(key):    # pop the list and iterate across it
                case = test.copy()
                case.update(item)
                generate(case, function)  # original test merged with each item
            return

    # For any arguments which are sequences, they are expanded into scalars
    for key in test:
        if key not in param['lists_to_not_expand']:
            if type(test[key]) in (tuple, list):
                for test[key] in test[key]:
                    generate(test, function)
                return

    # For integer arguments which are ranges (A..B[..C]), they are expanded
    for key in test:
        match = INT_RANGE_RE.match(str(test[key]))
        if match:
            for test[key] in xrange(int(match.group(1)),
                                    int(match.group(2))+1,
                                    int(match.group(3) or 1)):
                generate(test, function)
            return

    function(test)


if __name__ == '__main__':
    main()
