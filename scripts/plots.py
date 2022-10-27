import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns

def parse_test_results(folder: str):
    with open(f"../checkpoints/{folder}/test_results.json") as f:
        return json.load(f)[0]


def plot(ckpt_identifier: dict, name: str):
    """
    each entry should have: dataset names as keys
    which in turn should resolve to a list of ckpts in the order of loaded from (standard, robust, certifier) models
    """
    sns.set_context('poster') # Enlarges text size and linewidth for paper
    results_per_dataset = []
    for dataset in ckpt_identifier:
        results_per_setting = []
        for ckpt in ckpt_identifier[dataset]:
            results = parse_test_results(ckpt)
            results_per_setting.append(results)
        results_per_dataset.append(results_per_setting)

    plt.figure(figsize=(12, 5))
    ds = list(ckpt_identifier.keys())
    x = np.arange(len(ds))
    width = 0.1
    xlabels = list(ckpt_identifier.keys())
    plt.subplot(121)
    plt.title('Accuracy', fontsize=18)
    plt.bar(x-1.5*width, [results_per_dataset[_d][0]['test_acc'] for _d in x], width, label='standard')
    plt.bar(x-0.5*width, [results_per_dataset[_d][1]['test_acc'] for _d in x], width, label='robust')
    plt.bar(x+0.5*width, [results_per_dataset[_d][2]['test_acc'] for _d in x], width, label='certified')
    plt.bar(x+1.5*width, [results_per_dataset[_d][3]['test_acc'] for _d in x], width, label='constant')
    plt.xticks(x, xlabels, fontsize=15)

    plt.subplot(122)
    plt.title('Certificate', fontsize=18)
    plt.bar(x-1.5*width, [results_per_dataset[_d][0]['test_mean_delta'] for _d in x], width)
    plt.bar(x-0.5*width      , [results_per_dataset[_d][1]['test_mean_delta'] for _d in x], width)
    plt.bar(x+0.5*width, [results_per_dataset[_d][2]['test_mean_delta'] for _d in x], width)
    plt.bar(x+1.5*width, [results_per_dataset[_d][3]['test_mean_delta'] for _d in x], width)
    plt.yscale('log')
    plt.xticks(x, xlabels, fontsize=15)

    plt.figlegend() #loc='upper right', ncol=1, labelspacing=0.5, fontsize=14, bbox_to_anchor=(1.11, 0.9))
    # plt.tight_layout()
    plt.savefig(f'../plots/{name}.png')
    plt.savefig(f'../plots/{name}.pdf')
    plt.show()


def adaptation_results(ckpt_identifiers: dict):
    """
    ckpt_identifiers should have keys: {'covar_shift', 'metric_shift', 'spec_shift', 'covar_metric_shift'}
    each entry in itself should have: dataset names as keys
    which in turn should resolve to a list of ckpts in the order of loaded from (standard, robust, certifier) models
    """
    for k in ckpt_identifiers:
        print(k)
        plot(ckpt_identifiers[k], k)


if __name__ == '__main__':
    ckpt_identifiers = {
        'covar_shift': {
            'german': ['german_expt9', 'german_expt10', 'german_expt8', 'german_expt26'],
            # 'adult': ['adult_expt9', 'adult_expt10', 'adult_expt8', 'adult_expt26'],
            # 'credit': ['credit_expt9', 'credit_expt10', 'credit_expt8'],
        },
        'metric_shift': {'german': ['german_expt6', 'german_expt7', 'german_expt5', 'german_expt26'],
                         # 'adult': ['adult_expt6', 'adult_expt7', 'adult_expt5', 'adult_expt26'],
                         # 'credit': ['credit_expt6', 'credit_expt7', 'credit_expt5'],
                         },
        'spec_shift': {'german': ['german_expt16', 'german_expt19', 'german_expt17', 'german_expt26'],
                       # 'adult': ['adult_expt16', 'adult_expt19', 'adult_expt17', 'adult_expt26'],
                       # 'credit': ['credit_expt16', 'credit_expt119', 'credit_expt17'],
                       },
        'covar_metric_shift': {'german': ['german_expt12', 'german_expt13', 'german_expt11', 'german_expt26'],
                               # 'adult': ['adult_expt12', 'adult_expt13', 'adult_expt11', 'adult_expt26'],
                               # 'credit': ['credit_expt12', 'credit_expt13', 'credit_expt11'],
                               },
        'covar_metric_shift_full_dat': {'german': ['german_expt20', 'german_expt4', 'german_expt21', 'german_expt26']}
    }
    adaptation_results(ckpt_identifiers=ckpt_identifiers)
