import eval
import csv
import os
import argparse
import json
import subprocess
import torch

import urllib3

os.environ["HF_HUB_NUM_RETRIES"] = "0"

urllib3.util.retry.Retry.DEFAULT = urllib3.util.retry.Retry(0)


def load_json_from_file(filename):
    with open(filename, "r") as f:
        return json.load(f, strict=False)


def test_custom_attack(test_name, path_template, device, output_file):
    results = []
    for w in eval.WATERMARKS:
        results.append(eval.run_one(w, test_name, path_template.format(watermark=w), device))
    perf = [x['Watermark Performance'] for x in results]
    quality = [x['Quality Degradation'] for x in results]
    score = [x['Score'] for x in results]
    return (0, 'OK \n' + ', '.join([f'Watermark Performance: {sum(perf)/len(perf)}',
                                    f'Quality Degradation: {sum(quality)/len(quality)}',
                                    f'Score: {sum(score)/len(score)}']))

def run_tests(test_path, output_file):
    test_info = load_json_from_file(os.path.join(test_path, 'info.json'))
    test_name = test_info['attack']
    path_template = os.path.join(test_path, 'images', '{watermark}/')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if test_name == 'your':
        return test_custom_attack(test_name, path_template, device, output_file)
    
    cmd = (
        "python3 test.py "
        f"--device {device} "
        f"--attack {test_name} "
        f"--path-template {path_template} "
        f"--out {output_file}"
    )
    try:
        subprocess.run(f"({cmd}) 2>&1 | tee /dev/stderr", shell=True,
                       stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Command failed (code={e.returncode}). stderr: {e.stderr.strip()}")
    
    if not os.path.exists(output_file):
        raise Exception("Cannot find file with results")
    
    with open(output_file, 'r') as f:
        data = json.load(f)['per_watermark']
        failures = []
        errors = []
        for w in data:
            failures.extend(data[w][test_name]['failures'])
            errors.extend(data[w][test_name]['errors'])
        
        if len(errors) != 0 or len(failures) != 0:
            return (0, 'RE:\n' + '\n'.join(map(str, errors + failures)))
        return (3, 'OK')


def save_results(results, filename):
    header = ['score', 'conclusion']
    with open(filename, 'w', newline='') as resfile:
        writer = csv.writer(resfile)
        writer.writerow(header)
        resfile.flush()

        writer.writerow(results)
        resfile.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--test_dir', type=str, default='test_files')
    parser.add_argument('--output_file', type=str, default='results.csv')

    args = parser.parse_args()

    results = run_tests(args.test_dir, args.output_file)

    save_results(results, args.output_file)
    print('Results saved to: {}'.format(args.output_file))

