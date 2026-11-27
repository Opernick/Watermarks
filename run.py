#!/usr/bin/env python3

from json import dumps, load
from os import environ, makedirs
from os.path import join, abspath
from sys import argv, exit
from subprocess import run
import csv


def run_single_test(data_dir, output_dir):
    # '.'---user code
    #   |---additional code
    #   └---run.py
    # data_dir == dir with input data in tests

    cmds = [
        f'python3 test_all.py --test_dir={data_dir} --output_file={join(output_dir, "results.csv")}'
    ]

    for cmd in cmds:
        ret_code = run(cmd, shell=True).returncode
        if ret_code != 0:
            exit(ret_code)


def check_test(data_dir):
    # {data_dir}---output/results.csv
    #          \---gt/gt.csv

    with open(join(data_dir, 'output/results.csv')) as f:
        csvreader = csv.reader(f)
        results = list(csvreader)

    if len(results) == 2:
        result = results[1]  # Skip header, each result stored in separate test
        score = result[0]
        status = result[1]
        
        if status.startswith('OK \n'):
            score = ' \n'.join(status.split(' \n')[1:])
            status = 'OK'

    verdict = '\n'.join([status, score])  # OK 0.9 or RE 0.0
    
    if environ.get('CHECKER'):
        if status != 'OK':
            raise Exception(status)
         
        print(verdict)
    return verdict


def grade(data_path):
    # {data_dir}---results.json

    results = load(open(join(data_path, 'results.json')))
    mark = 0
    for result in results:
        if "Time limit" in result['status']:
            continue
        
        spl = result['status'].split()
        if len(spl) == 0 or spl[0] != 'OK':
            continue
        
        try:
            mark += float(spl[1])
        except ValueError:
            pass
    
    metric = results[-1]['status'].split()[1:] if results[-1]['status'].startswith('OK\n') else ['']
    res = {'description': " ".join(metric), 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            # Run each test
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            # Put a mark for each test result
            check_test(data_dir)
        elif mode == 'grade':
            # Put overall mark
            grade(data_dir)
