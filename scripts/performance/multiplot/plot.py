#!/usr/bin/env python3

import argparse
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import yaml
from pathlib import Path

def plot_data(gflops_dicts, const_args_dicts, funcname_list, machine_spec_dict, savedir, arch, theo_max, size_arg = 'N'):
    """
    plots gflops data from dictionaries, one plot for each common precision present in all dictionaries.

    Parameters:
        gflops_dicts (list[dict{string: list[(int, float)]}]): data as given by :func:`get_data_from_directories`.
        funcname_list (list[string]): a list of funcname for each data set to be plotted and used as a savefile name.
        savedir (string): directory where resulting plots will be saved.
        size_arg (string): x axis title on plot.
    """
    if len(gflops_dicts) == 0:
        return

    gflops_dict0 = gflops_dicts[0]
    for prec, _ in gflops_dict0.items():
        colors=iter(cm.rainbow(np.linspace(0,1,len(gflops_dicts))))
        figure, axes = plt.subplots(figsize=(7,7))

        for gflops_dict, funcname, const_args_dict in zip(gflops_dicts, funcname_list, const_args_dicts):
            cur_color = next(colors)
            if prec not in gflops_dict:
                continue
            gflops = gflops_dict[prec]
            gflops.append((0, 0)) # I prefer having a 0 at the bottom so the performance looks more accurate
            sorted_tuples = sorted(gflops)
            sorted_sizes = [x[0] for x in sorted_tuples]
            sorted_gflops = [x[1] for x in sorted_tuples]

            if(prec == "f32_r"):
                function_label = "s" + funcname
            elif(prec == "f64_r"):
                function_label = "d" + funcname
            elif(prec == "f32_c"):
                function_label = "c" + funcname
            elif(prec == "f64_c"):
                function_label = "z" + funcname

            if(theo_max == True):
                theo_max_value = machine_spec_dict[function_label]
                sorted_gflops[:] = [gf / theo_max_value for gf in sorted_gflops]

            function_label = function_label + " :  " + const_args_dict[prec]

            axes.scatter(sorted_sizes, sorted_gflops, color=cur_color, label=function_label)
            axes.plot(sorted_sizes, sorted_gflops, '-o', color=cur_color)

        if(theo_max == True):
            axes.set_ylim(0, 1)
            axes.set_ylabel('gflops / theoretical_maximum_gflops')
        else:
            axes.set_ylabel('gflops')

        axes.set_xlabel('='.join(size_arg)) # in case we add multiple params

        # magic numbers from performancereport.py to make plots look nice
        axes.legend(fontsize=10, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode='expand', borderaxespad=0.)
        figure.tight_layout(rect=(0,0.05,1.0,0.94))

        figure.suptitle(arch, y=0.96)

        filename = ''
        for funcname in funcname_list:
            if filename != '':
                filename += '_'
            filename += funcname
        filename += '_' + prec
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figure.savefig(os.path.join(os.getcwd(), savedir, filename))


def get_function_name(filename):
    function_str = "function"
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()
    else:
        print(filename + "does not exist")
    for i in range(0, len(lines)):
        if(function_str in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            function_idx = arg_line.index(function_str)
            return data_line[function_idx]

def get_arch_from_yaml(machine_spec_yaml_file):
    machine_spec_dict = yaml.safe_load(Path(machine_spec_yaml_file).read_text())
    return machine_spec_dict['arch']

def get_data_from_file(filename, output_param='rocblas-Gflops', xaxis_str1='N', xaxis_str2='M', yaxis_str='rocblas-Gflops'):

    precision_str = "compute_type"
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()

    function_name = get_function_name(filename)

    cur_dict = {}
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            if(xaxis_str1 in arg_line):
                xaxis_idx = arg_line.index(xaxis_str1)
            if(xaxis_str2 in arg_line):
                xaxis_idx = arg_line.index(xaxis_str2)
            yaxis_idx = arg_line.index(yaxis_str)
            size_perf_tuple = (int(data_line[xaxis_idx]), float(data_line[yaxis_idx]))

            precision_idx = arg_line.index(precision_str)
            precision = data_line[precision_idx]
            if precision in cur_dict:
                cur_dict[precision].append(size_perf_tuple)
            else:
                cur_dict[precision] = [size_perf_tuple]

    return cur_dict

tracked_param_list = [ 'transA', 'transB', 'uplo', 'diag', 'side', 'M', 'N', 'K', 'KL', 'KU', 'alpha', 'alphai', 'beta', 'betai',
                       'incx', 'incy', 'lda', 'ldb', 'ldd', 'stride_x', 'stride_y', 'stride_a', 'stride_b', 'stride_c', 'stride_d',
                       'batch_count']

# return string of arguments that remain constant. For example, transA, transB, alpha, beta, incx may remain
# constant. By contrast, M, N, K, lda, ldb, ldc may change
#def get_const_args_str(filename, output_param='rocblas-Gflops'):
def get_const_args_dict(filename, output_param='rocblas-Gflops'):

    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()


    precision_str = "compute_type"
    precisions = []
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])

            precision_idx = arg_line.index(precision_str)
            precision = data_line[precision_idx]
            if precision not in precisions:
                precisions.append(precision)

    const_args_dict = {}

    for precision in precisions:

        function_param_list = tracked_param_list

        arg_line_index_dict = {}
        arg_line_value_dict = {}
        for i in range(0, len(lines)):
            if((output_param in lines[i]) and (precision in lines[i+1])):
                arg_line = lines[i].split(",")
                data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])

                if not arg_line_index_dict:
                    for arg in arg_line :
                        if(arg in function_param_list):
                            index = arg_line.index(arg)
                            value = data_line[index]
                            arg_line_index_dict[arg]=index
                            arg_line_value_dict[arg]=value
                else:
                    for arg in arg_line :
                        if(arg in function_param_list):
                            index = arg_line.index(arg)
                            value = data_line[index]
                            previous_value = arg_line_value_dict[arg]
                            if(value != previous_value):
                                function_param_list.remove(arg)
                                del arg_line_value_dict[arg]

        const_args_str = ""
        for key, value in arg_line_value_dict.items():
            if(const_args_str == ""):
                const_args_str += key + "=" + value
            else:
                const_args_str += ", " + key + "=" + value

        const_args_dict[precision] = const_args_str
    return const_args_dict

if __name__ =='__main__':

    parser = argparse.ArgumentParser(
            description='plot rocblas-bench results for multiple csv files',
            epilog='Example usage: python3 plot_benchmarks.py ' +
                    '-l blas1 -t gfx906  -f scal -f axpy  --label1 "N" --label2 "M"')

    parser.add_argument('-l', '--level',      help='BLAS level',          dest='level',          default='blas1')
    parser.add_argument('-t',   '--tag',      help='tag',                 dest='tag',            default='gfx906')
    parser.add_argument(     '--label1',      help='label1',              dest='label1',         default='N')
    parser.add_argument(     '--label2',      help='label2',              dest='label2',         default='M')
    parser.add_argument('-f'           ,      help='function name',       dest='function_names', required=True, action='append')
    parser.add_argument(     '--theo_max',    help="perf vs theo_max",    dest='theo_max', default="false", action='store_true')
    parser.add_argument(     '--no_theo_max', help="no perf vs theo_max", dest='theo_max', action='store_false')
    parser.set_defaults(theo_max=False)

    args = parser.parse_args()

    funcname_list = []

    res_dicts = []
    const_args_dicts = []

    const_args_list = []

    output_dir = os.path.join(args.level,args.tag)

    if (args.theo_max == True):
        output_dir = os.path.join(output_dir, "plots_vs_theo_max")
    else:
        output_dir = os.path.join(output_dir, "plots_gflops")

    machine_spec_yaml_file = os.path.join(args.level, args.tag, "machine_spec.yaml")

    machine_spec_dict = yaml.safe_load(Path(machine_spec_yaml_file).read_text())
    arch = machine_spec_dict['arch']

    for function_name in args.function_names:

        output_filename = os.path.join(args.level, args.tag, function_name+".csv")

        cur_dict = get_data_from_file(output_filename, "rocblas-Gflops", args.label1, args.label2, "rocblas-Gflops")

        res_dicts.append(cur_dict)

        const_args_dict = get_const_args_dict(output_filename, "rocblas-Gflops")

        const_args_dicts.append(const_args_dict)

        function_name = get_function_name(output_filename)
        funcname_list.append(function_name)

    print("plotting for: ", funcname_list)
    plot_data(res_dicts, const_args_dicts, funcname_list, machine_spec_dict, output_dir, arch, args.theo_max, args.label1)
