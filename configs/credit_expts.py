from utils import get_best_ckpt

OUT_DIR = 'checkpoints'
SMALL_DATA_FRAC = 0.5  # (0.0028)
DEFAULT_SHIFT_FACTOR = 1.
DEFAULT_LABEL_SHIFT_FACTOR = 0.5

DATA_FRACS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1.]

"""
expts description
1. standard model, 100% data
2. certifier model, 100% data
3. adversarially trained model, 100% data
---------
metric shifts (LP->SENSR), 10% data
---------
5. load from certifier model, 10% data
6. load from standard model, 10% data
7. load from robust model, 10% data
27. random initialized, 10% data
---------
Covariate shift (shift factor=1.), 10% data
---------
8. load from certifier model
9. load from standard model
10. load from robust model
28. random initialized
---------
Metric (LP->SENSR) + covariate shift (shift factor=1.)
---------
======
10% data
======
11. load from certifier model
12. load from standard model
13. load from robust model
29. random initialized
======
100% data
======
4. load from certifier model, full data
20. load from standard model, full data
21. load from robust model, full data
30. random initialized
---------
Spec shift (new sens attribute 'age'), SENSR metric, using sens attribute
---------
=======
100% data
=======
14. Standard model, 100% data 
15. Certifier model, 100% data 
18. Robust model, 100% data
========
10% data
========
16. load from standard model, 10% data, new sens attribute
17. load from certifier model, 10% data, new sens attribute
19. load from robust model, 10% data, new sens attribute
31. random initialized
---------
Sanity check pretrained models without training on shifted datasets
Metric + Covar shift 
---------
23. loaded from standard model (1.) 
24. loaded from certified model (2.)
25. loaded from robust model (3.)
---------
Sanity check II
---------
26. Constant classifier, should serve any setting where label distribution did not change
---------
Label shift
---------
32. load from standard model, 10% data
33. load from certifier model, 10% data
34. load from robust model, 10% data
---------
Training data size
---------
35_{training_data_proportion}. spec shift: load from certified model (similar to 11)
36_{training_data_proportion}. cov shift: load from certified model (similar to 8)
"""

credit_expts = {
    # =======================================================================================
    # Normal Training
    # Double Checked
    'credit_expt0': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 0.0,
            'alpha': 0.0,
        }
    },
    # F-IBP-LP training
    # Double Checked
    'credit_expt100': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'trainer_kwargs': {
            # 'epsilon': 0.75,
            'epsilon': 1.0,
            'alpha': 0.075,
        }
    },
    # F-IBP-SENSR training
    # Double Checked
    'credit_expt101': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.035,
        }
    },
    # R-IBP training
    # Double Checked
    'credit_expt102': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'robust',
        'metric': 'LP',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.025,
        }
    },
    # F-PGD training
    # Double Checked
    'credit_expt103': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'LP',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 0.5,
            'alpha': 0.025,
        }
    },
    # metric shift + data fraction control
    # 0.1 is the smallest fraction we can set without going below 50 examples on credit dataset

    # Retrain to LP from standard
    'credit_expt104': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt0"),
        'trainer_kwargs': {
            'epsilon': 1.0,
            'alpha': 0.15,
        }
    },
    # Retrain to SENSR from standard
    'credit_expt105': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt0"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.1,
        }
    },

    #################
    # Specification shift eg. lp -> sensr or sensr -> lp
    #################
    # Retrain to LP from R-IBP
    'credit_expt106': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt101"),
        'trainer_kwargs': {
            'epsilon': 1.0,
            'alpha': 0.15,
        }
    },
    
    'credit_expt107': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt102"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.1,
        }
    },
    
    'credit_expt108': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt100"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.1,
        }
    },

    # Retrain to SENSR from R-IBP
    'credit_expt109': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt101"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.1,
        }
    },

    # F-PGD training
    'credit_expt110': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'SENSR',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.05,
        }
    },

    # Retrain from LP-PGD to SENSR
    'credit_expt1101': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'SENSR',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt110"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.1,
        }
    },

    # Retrain from SENSR-PGD to LP
    'credit_expt1102': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'LP',
        'data_frac': SMALL_DATA_FRAC,
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt103"),
        'trainer_kwargs': {
            'epsilon':  1.0,
            'alpha': 0.05,
        }
    },

    ###################
    # Covar shift
    ###################

    # Standard -> F-IBP-LP (shifted)
    'credit_expt111': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': SMALL_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.1,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt0"),
    },

    # R-IBP -> F-IBP-LP (shifted)
    'credit_expt112': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': SMALL_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.1,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt102"),
    },

    # F-IBP-LP -> F-IBP-LP (shifted)
    'credit_expt113': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': SMALL_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.1,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt100"),
    },

    ###################
    # label shift
    ###################

    # Standard -> F-IBP-LP (shifted)
    'credit_expt114': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': 2 * SMALL_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.5,
            'alpha': 0.15,  # 0.15 (92.1, 0.365) 0.3 (90.3, 0.61) 0.5 (87.4, 0.20) 0.7 (trivial accuracy)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt0"),
    },

    # R-IBP -> F-IBP-LP (shifted)
    'credit_expt115': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': 2 * SMALL_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.5,
            'alpha': 0.5,  # 0.15 (92.4, 0.32) 0.3 (91, 0.03) 0.5 (91.0, 0.015)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt102"),
    },

    # F-IBP-LP -> F-IBP-LP (shifted)
    'credit_expt116': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': 2 * SMALL_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.5,
            'alpha': 0.15,  # 0.5 (82.17, 0.2) 0.3 (91.5, 0.27) 0.15 (92.4, 0.06)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt100"),
    },

    #############
    # data fraction experiments with covar shift
    #############
    'credit_datafrac_covshift_template': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.1,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt100"),
    },
    
    # Training at different scales
    'credit_modelscale_template': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 0.25, # was 1.0 and then was 0.05
            'alpha': 0.075,
        },
        'model_kwargs': {
            'hidden_lay': 1,
            'hidden_dim': 128,
        },
    },
    
    'credit_expt119': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': .5,
            'alpha': 0.0125, # 0.025 is big
        },
    },
    'credit_expt120': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': .2,
            'alpha': 0.1,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt102"),
    },

    'credit_expt121': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt0"),
        'trainer_kwargs': {
            'epsilon': 7.5e-3,
            'alpha': 0.01, # 0.05 (trivial accuracy), 0.025 (trivial) 0.01 (62.1, 0.33)
        }
    },
    'credit_expt122': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 7.5e-3,
            'alpha': 0.01, # 0.025 (50.2, 0.09), 0.1 (62.1, 0.14)
        }
    },
    'credit_expt123': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'credit',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt102"),
        'trainer_kwargs': {
            'epsilon': 7.5e-3,
            'alpha': 0.0125, # 0.025 (57.5, 0.22) 0.0125 (62, 0.21) 0.018 (62.1, 0.21)
        }
    },
    
    # Retrain to LP from SENSR
    'credit_expt125': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'unlabeled',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': SMALL_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.1,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt102"),
    },
    
    'credit_expt127': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'dataless',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': 1.0,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.1,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt102"),
    },
    
    'credit_expt128': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'robust',
        'metric': 'LP',
        'dataset': 'credit',
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.0125,
        },
        'model_kwargs': {
            'hidden_dim': 128,
            'hidden_lay': 2,
        },
        'use_weighted_ce': True,
    },
    
    'credit_expt129': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': 2*SMALL_DATA_FRAC,
        'num_adapt': 1,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.0025,
        },
        'model_kwargs': {
            'hidden_dim': 128,
            'hidden_lay': 2,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt128"),
        'max_steps': 15000,
        'use_weighted_ce': True,
    },
    
    'credit_expt130': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'trainer_kwargs': {
            # 'epsilon': 0.75,
            'epsilon': 1.0,
            'alpha': 0.075,
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
    },
    
    'credit_expt131': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'credit',
        'num_adapt': 1,
        'data_frac': 2 * SMALL_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.5,
            'alpha': 0.5,  # 0.15 (92.4, 0.32) 0.3 (91, 0.03) 0.5 (91.0, 0.015)
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt130"),
    },
    
    'credit_expt132': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'unlabeled',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': SMALL_DATA_FRAC,
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.35,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt101"),
    },
    
    'credit_expt133': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'dataless',
        'metric': 'LP',
        'dataset': 'credit',
        'data_frac': 1.0,
        'trainer_kwargs': {
            'epsilon': 0.2,
            'alpha': 0.225,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/credit_expt101"),
    },
    
}

for data_frac in DATA_FRACS:
    _d = credit_expts['credit_datafrac_covshift_template'].copy()
    _d['data_frac'] = data_frac
    credit_expts[f'credit_expt117_{data_frac:0.3f}'] = _d
    
WIDTHS = [8, 12, 16, 24, 64, 128, 256, 512]
DEPTHS = [1, 2, 3]
import copy
for depth in DEPTHS:
    for width in WIDTHS:
        _d = copy.deepcopy(credit_expts['credit_modelscale_template'])
        _d['model_kwargs']['hidden_dim'] = width
        _d['model_kwargs']['hidden_lay'] = depth
        credit_expts[f'credit_expt118_%s_%s'%(depth, width)] = _d
        del(_d)
        

