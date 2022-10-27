from utils import get_best_ckpt
OUT_DIR = 'checkpoints'
DEFAULT_SHIFT_FACTOR = 1.
DEFAULT_LABEL_SHIFT_FACTOR = 0.25
SMALLEST_DATA_FRAC = 0.2

DATA_FRACS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1.]

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
german_expts = {
    # =======================================================================================
    # Normal Training
    # Double Checked
    'german_expt0': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 0.0,
            'alpha': 0.0,
        }
    },
    # F-IBP-LP training
    # Double Checked
    'german_expt100': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 0.4,
            'alpha': 0.05,
        }
    },
    # F-IBP-SENSR training
    # Double Checked
    'german_expt101': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 70e-4,
            'alpha': 0.02,
        },
        'max_steps': 35000,
    },  
    # R-IBP training
    # Double Checked
    'german_expt102': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'robust',
        'metric': 'LP',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.025,
        }
    }, 
    # F-PGD training
    # Double Checked
    'german_expt103': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'LP',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 0.4,
            'alpha': 0.05,
        }
    }, 

    #################
    # Can we adapt standard model for fairness using very small data?
    #################
    # Retrain to LP from standard
    'german_expt104': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt0"),
        'trainer_kwargs': {
            'epsilon': 0.4,
            'alpha': 0.05,
        }
    },
    # Retrain to SENSR from standard
    'german_expt105': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt0"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.05,
        }
    },

    #################
    # Specification shift eg. lp -> sensr or sensr -> lp
    #################
    # Retrain to LP from R-IBP
    'german_expt106': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt101"),
        'trainer_kwargs': {
            'epsilon': 0.4,
            'alpha': 0.025,
        }
    },

    # Retrain to SENSR from R-IBP
    'german_expt107': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt102"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.0125, # was 0.025
        }
    },

    # Retrain to SENSR from LP
    'german_expt108': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt100"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.0125,
        }
    },

    # Retrain to LP from SENSR
    'german_expt109': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt101"),
        'trainer_kwargs': {
            'epsilon': 0.4,
            'alpha': 0.0125,
        }
    },

    # F-PGD training (SENSR)
    'german_expt110': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'SENSR',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.0125,
        }
    },
    
    # ! Missing config (sensr -> cert LP)
    'german_expt1101': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt101"),
        'trainer_kwargs': {
            'epsilon': 0.4,
            'alpha': 0.025,
        }
    },
    
    # ! Missing config (sensr -> cert sensr)
    'german_expt1102': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt103"),
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.025,
        }
    },
   
    ###################
    # Covar shift
    ###################

    # Standard -> F-IBP-LP (shifted)
    'german_expt111': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.0125,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt0"),
    },

    # R-IBP -> F-IBP-LP (shifted)
    'german_expt112': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.0125,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt102"),
    },

    # F-IBP-LP -> F-IBP-LP (shifted)
    'german_expt113': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.0125,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt100"),
    },

    ###################
    # label shift
    ###################

    # Standard -> F-IBP-LP (shifted)
    'german_expt114': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.02,
            'alpha': 0.2, # 0.0 (84.5, 0.597) 0.05 (85.5, 0.512) 0.1 (83, 0.423) 0.2 (84, 0.38) 0.4 (trivial acc)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt0"),
    },

    # R-IBP -> F-IBP-LP (shifted)
    'german_expt115': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.02,
            'alpha': 0.4, # 0.1 (85, 0.38) 0.2 (82.6, 0.2) 0.4 (0.766, 0.004)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt102"),
    },

    # F-IBP-LP -> F-IBP-LP (shifted)
    'german_expt116': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.02,
            'alpha': 0.3, # 0.2 (0.76, 0.1) 0.3 (0.766, 0.032)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt100"),
    },

    #############
    # data fraction experiments with covar shift
    #############
    'german_datafrac_covshift_template': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.025, # was 0.0125
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt100"),
    },
    
    # Training at different scales
    'german_modelscale_template': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.05,
        },
        'model_kwargs': {
            'hidden_lay': 1,
            'hidden_dim': 128,
        },
    },
    'german_expt119': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.0125,
        },
    },
    'german_expt120': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.01, # 0.025 (trivial accuracy), 0.0125 (good cert, but bad accuracy)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt102"),
    },

    'german_expt121': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt0"),
        'trainer_kwargs': {
            'epsilon': 7.5e-3,
            'alpha': 0.2, # 0.4 (53.9, 0.4 (test with eps=1e-3)) 0.2 (58.2, 0.45 (test with eps=1e-3)) 0.1 (55.8, 0.99, 0.41 (test with eps=1e-3)) 0.05 (54.2, 0.99)
        }
    },
    'german_expt122': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 7.5e-3,
            'alpha': 0.0035, # with eps=1e-3{0.025 (64.9, 0.61) 0.05 (51.6, 0.12) 0.04 (50.7, 0.38)}
            # with eps=7.5e-3{0.0125, 0.01 (trivial accuracy), 0.005 (54.9, 0.17), 0.001 (68.3, 0.94) 0.0025 (62.7, 0.48) 0.0035 (64.2, 0.29)}
        }
    },
    'german_expt123': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'german',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt102"),
        'trainer_kwargs': {
            'epsilon': 7.5e-3,
            'alpha': 0.0009, # 0.0035 (63.5, 0.59) 0.005 (62.5, 0.7) 0.01 (53.2, 0.3) 0.007 (64.5, 0.54) 0.009 (54.7, 0.25) 0.008 (59.2, 0.45) 0.0125 (51.6)
        }
    },
    
    
    # These must be indicative of some error.
    'german_expt125': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'unlabeled',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.0125,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt102"),
    },
    # This is definitely indicative of an error...
    'german_expt127': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'dataless',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 1.0,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.0125,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt102"),
    },
    
    'german_expt128': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'robust',
        'metric': 'LP',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 0.01,
            'alpha': 0.001,
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
        #'use_weighted_ce': True,
    }, 

    'german_expt129': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'num_adapt': 1,
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.025,
            'alpha': 0.00025,
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt128"),
        'max_steps': 15000,
        #'use_weighted_ce': True,
    },
    
    
    'german_expt130': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'trainer_kwargs': {
            'epsilon': 0.4,
            'alpha': 0.05,
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
    },
    'german_expt131': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'german',
        'num_adapt': 1,
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.02,
            'alpha': 0.4, # 0.1 (85, 0.38) 0.2 (82.6, 0.2) 0.4 (0.766, 0.004)
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt130"),
    },
    
    
    
    # These must be indicative of some error.
    'german_expt132': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'unlabeled',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.175,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt101"),
    },
    # This is definitely indicative of an error...
    'german_expt133': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'dataless',
        'metric': 'LP',
        'dataset': 'german',
        'data_frac': 1.0,
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.225,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/german_expt101"),
    },
    
    
    
    
}

for data_frac in DATA_FRACS:
    _d = german_expts['german_datafrac_covshift_template'].copy()
    _d['data_frac'] = data_frac
    german_expts[f'german_expt117_{data_frac:0.3f}'] = _d

WIDTHS = [8, 12, 16, 24, 64, 128, 256, 512, 1024, 2048]
DEPTHS = [1, 2, 3]
import copy
for depth in DEPTHS:
    for width in WIDTHS:
        _d = copy.deepcopy(german_expts['german_modelscale_template'])
        _d['model_kwargs']['hidden_dim'] = width
        _d['model_kwargs']['hidden_lay'] = depth
        german_expts[f'german_expt118_%s_%s'%(depth, width)] = _d
        del(_d)
        
        