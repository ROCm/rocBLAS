#!/usr/bin/python
"""Expand rocBLAS YAML test data file into binary Arguments records"""

import re
import sys
import os
import argparse
import ctypes
from fnmatch import fnmatchcase
try:  # Import either the C or pure-Python YAML parser
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import yaml

# Regex for type names in the YAML file. Optional *nnn indicates array.
TYPE_RE = re.compile(r'[a-z_A-Z]\w*(:?\s*\*\s*\d+)?$')

# Regex for integer ranges A..B[..C]
INT_RANGE_RE = re.compile(r'\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*(?:\.\.\s*(-?\d+)\s*)?$')

# Regex for include: YAML extension
INCLUDE_RE = re.compile(r'include\s*:\s*(.*)')

args = {}
testcases = set()
datatypes = {}
param = {}


def main():
    (infile, args['outfile'], args['includes']) = parse_args()
    for doc in get_yaml_docs(infile):
        process_doc(doc)


def process_doc(doc):
    """Process one document in the YAML file"""

    # Ignore empty documents
    if not doc or not doc.get('Tests'):
        return

    # Clear datatypes and params from previous documents
    datatypes.clear()
    param.clear()

    # Return dictionary of all known datatypes
    datatypes.update(get_datatypes(doc))

    # Arguments structure corresponding to C/C++ structure
    param['Arguments'] = type('Arguments', (ctypes.Structure,),
                              {'_fields_': get_arguments(doc)})

    # Special names which get expanded as lists of arguments
    param['dict_lists_to_expand'] = doc.get('Dictionary lists to expand') or ()

    # Lists which are not expanded
    param['lists_to_not_expand'] = doc.get('Lists to not expand') or ()

    # Defaults
    defaults = doc.get('Defaults') or {}

    # Known Bugs
    param['known_bugs'] = doc.get('Known bugs') or []

    # Instantiate all of the tests, starting with defaults
    for test in doc['Tests']:
        case = defaults.copy()
        case.update(test)
        generate(case, instantiate)


def parse_args():
    """Parse command-line arguments, returning input and output files"""
    parser = argparse.ArgumentParser(description="""
Expand rocBLAS YAML test data file into binary Arguments records
""")
    parser.add_argument('infile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('-o', '--out',
                        dest='outfile',
                        type=argparse.FileType('wb'),
                        default=sys.stdout)
    parser.add_argument('-I',
                        help="Add include path",
                        action='append',
                        dest='includes',
                        default=[])
    parsed = parser.parse_args()
    return (parsed.infile, parsed.outfile, parsed.includes)


def read_yaml_file(file):
    """Read the YAML file, processing include: lines as an extension"""
    file_dir = os.path.dirname(file.name) or os.getcwd()
    source = []
    for line_no, line in enumerate(file, start=1):
        # Keep track of file names and line numbers for each line of YAML
        match = line.startswith('include') and INCLUDE_RE.match(line)
        if not match:
            source.append([line, file.name, line_no])
        else:
            include_file = match.group(1)
            include_dirs = [file_dir] + args['includes']
            for path in include_dirs:
                path = os.path.join(path, include_file)
                if os.path.exists(path):
                    source.extend(read_yaml_file(open(path, 'r')))
                    break
            else:
                sys.exit("In file " + file.name + ", line " +
                         str(line_no) + ", column " + str(match.start(1)+1) +
                         ":\n" + line.rstrip() + "\n" + " " * match.start(1) +
                         "^\nCannot open " + include_file +
                         "\n\nInclude paths:\n" + "\n".join(include_dirs))
    file.close()
    return source


def get_yaml_docs(infile):
    """Parse the YAML file"""
    source = read_yaml_file(infile)
    source_str = ''.join([line[0] for line in source])

    def mark_str(mark):
        line = source[mark.line]
        return("In file " + line[1] + ", line " + str(line[2]) + ", column " +
               str(mark.column + 1) + ":\n" + line[0].rstrip() + "\n" +
               ' ' * mark.column + "^\n")

    # We iterate through all of the documents to properly diagnose errors,
    # because the load_all generator does not handle exceptions correctly.
    docs = []
    load = Loader(source_str)
    while load.check_data():
        try:
            doc = load.get_data()
        except yaml.YAMLError as err:
            sys.exit((mark_str(err.problem_mark) if err.problem_mark else "") +
                     (err.problem + "\n" if err.problem else "") +
                     (err.note + "\n" if err.note else ""))
        else:
            docs.append(doc)
    return docs


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
                        dt[subtype] = eval(name+'.'+subtype, dt)
            elif isinstance(decl, str) and TYPE_RE.match(decl):
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
    if test['transA'] == '*' or test['transB'] == '*':
        test.setdefault('lda', 0)
        test.setdefault('ldb', 0)
        test.setdefault('ldc', 0)
        test.setdefault('ldd', 0)
    else:
        test.setdefault('lda', test['M'] if test['transA'].upper() == 'N' else
                        test['K'])
        test.setdefault('ldb', test['K'] if test['transB'].upper() == 'N' else
                        test['N'])
        test.setdefault('ldc', test['M'])
        test.setdefault('ldd', test['M'])
        if test['batch_count'] > 0:
            test.setdefault('stride_a', test['lda'] *
                            (test['K'] if test['transA'].upper() == 'N' else
                             test['M']))
            test.setdefault('stride_b', test['ldb'] *
                            (test['N'] if test['transB'].upper() == 'N' else
                             test['K']))
            test.setdefault('stride_c', test['ldc'] * test['N'])
            test.setdefault('stride_d', test['ldd'] * test['N'])
            return

    test.setdefault('stride_a', 0)
    test.setdefault('stride_b', 0)
    test.setdefault('stride_c', 0)
    test.setdefault('stride_d', 0)


def write_signature(out):
    """Write the signature used to verify binary file compatibility"""
    if 'signature_written' not in args:
        sig = 0
        byt = bytearray("rocBLAS")
        byt.append(0)
        last_ofs = 0
        for (name, ctype) in param['Arguments']._fields_:
            member = getattr(param['Arguments'], name)
            for i in range(0, member.offset - last_ofs):
                byt.append(0)
            for i in range(0, member.size):
                byt.append(sig ^ i)
            sig = (sig + 89) % 256
            last_ofs = member.offset + member.size
        for i in range(0, ctypes.sizeof(param['Arguments']) - last_ofs):
            byt.append(0)
        byt.extend(bytearray("ROCblas"))
        byt.append(0)
        out.write(byt)
        args['signature_written'] = True


def write_test(test):
    """Write the test case out to the binary file if not seen already"""

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
        for (name, ctype) in param['Arguments']._fields_
    ))

    byt = bytearray(arg)
    sig = tuple(byt)
    if sig not in testcases:
        testcases.add(sig)
        write_signature(args['outfile'])
        args['outfile'].write(byt)


def instantiate(test):
    """Instantiate a given test case"""
    test = test.copy()

    # Any Arguments fields declared as enums
    enum_args = [decl[0] for decl in param['Arguments']._fields_
                 if decl[1].__module__ == '__main__']
    try:
        setdefaults(test)
        # For enum arguments, replace name with value
        for typename in enum_args:
            test[typename] = datatypes[test[typename]]

        # Match known bugs
        if test['category'] not in ('known_bug', 'disabled'):
            for bug in param['known_bugs']:
                for key, value in bug.items():
                    if key not in test:
                        break
                    if key == 'function':
                        if not fnmatchcase(test[key], value):
                            break
                    # For keys declared as enums, compare resulting values
                    elif test[key] != (datatypes.get(value) if key in enum_args
                                       else value):
                        break
                else:  # All values specified in known bug match test case
                    test['category'] = 'known_bug'
                    break

        write_test(test)

    except KeyError as err:
        sys.exit("Undefined value " + str(err) + "\n" + str(test))


def generate(test, function):
    """Generate test combinations by iterating across lists recursively"""
    test = test.copy()

    # For specially named lists, they are expanded and merged into the test
    # argument list. When the list name is a dictionary of length 1, its pairs
    # indicate that the argument named by its key takes on values paired with
    # the argument named by its value, which is another dictionary list. We
    # process the value dictionaries' keys in alphabetic order, to ensure
    # deterministic test ordering.
    for argname in param['dict_lists_to_expand']:
        if type(argname) == dict:
            if len(argname) == 1:
                arg, target = argname.items()[0]
                if arg in test and type(test[arg]) == dict:
                    pairs = sorted(test[arg].items(), key=lambda x: x[0])
                    for test[arg], test[target] in pairs:
                        generate(test, function)
                    return
        elif argname in test and type(test[argname]) in (tuple, list, dict):
            # Pop the list and iterate across it
            ilist = test.pop(argname)

            # For a bare dictionary, wrap it in a list and apply it once
            for item in [ilist] if type(ilist) == dict else ilist:
                case = test.copy()
                case.update(item)
                generate(case, function)  # original test merged with each item
            return

    for key in sorted(test.keys()):
        # Integer arguments which are ranges (A..B[..C]) are expanded
        if type(test[key]) == str:
            match = INT_RANGE_RE.match(str(test[key]))
            if match:
                for test[key] in range(int(match.group(1)),
                                       int(match.group(2))+1,
                                       int(match.group(3) or 1)):
                    generate(test, function)
                return

        # For sequence arguments, they are expanded into scalars
        elif (type(test[key]) in (tuple, list) and
              key not in param['lists_to_not_expand']):
            for test[key] in test[key]:
                generate(test, function)
            return

    function(test)


if __name__ == '__main__':
    main()
