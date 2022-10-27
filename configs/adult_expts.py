from utils import get_best_ckpt
OUT_DIR = 'checkpoints'
SMALLEST_DATA_FRAC = 0.2 # (>0.0021)
DATA_FRACS = [ 0.003, 0.006, 0.009, 0.012, 0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1.]
DEFAULT_SHIFT_FACTOR = 1.
DEFAULT_LABEL_SHIFT_FACTOR = 0.25

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
adult_expts = {
    # =======================================================================================
    # Normal Training
    # Double Checked
    'adult_expt0': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'trainer_kwargs': {
            'epsilon': 0.0,
            'alpha': 0.0,
        }
    },
    # Double Checked
    'adult_expt100': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'trainer_kwargs': {
            #'epsilon': 0.1,
            'epsilon': 0.25,
            'alpha': 0.05,
        }
    },
    # F-IBP-SENSR training
    # Double Checked
    'adult_expt101': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'adult',
        'trainer_kwargs': {
            'epsilon': 75e-4,
            'alpha': 0.045,
        }
    },
    # R-IBP training
    # Double Checked
    'adult_expt102': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'robust',
        'metric': 'LP',
        'dataset': 'adult',
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.05,
        }
    },
    # F-PGD training
    # Double Checked
    'adult_expt103': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'LP',
        'dataset': 'adult',
        'trainer_kwargs': {
            'epsilon': 0.25,
            'alpha': 0.05,
        }
    },
    #################
    # Can we adapt standard model for fairness using very small data?
    #################
    # Double Checked
    'adult_expt104': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt0"),
        'trainer_kwargs': {
            'epsilon': 0.25,
            'alpha': 0.2,
        },
        'max_steps': 5000
    },
    # Retrain to SENSR from standard
    # if using same epsilon-alpha of similar experiments: 107, 107, we get poor performance (~50%) and poor certificate.
    # These values are set such that we get at least good accuracy
    # Double Checked
    'adult_expt105': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt0"),
        'trainer_kwargs': {
            'epsilon': 7.5e-2,
            'alpha': 0.0175, #was 0.01
        },
    },

    #################
    # Specification shift eg. lp -> sensr or sensr -> lp
    #################
    # Retrain to LP from R-IBP
    'adult_expt106': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt101"),
        'trainer_kwargs': {
            'epsilon': 0.25,
            'alpha': 0.5, # was 0.4
        }
    },

    # Retrain to SENSR from R-IBP
    'adult_expt107': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt102"),
        'trainer_kwargs': {
            'epsilon': 7.5e-2, 
            'alpha': 0.0165, #was 0.01 (too small) then 0.025 (too large)
        }
    },

    # Retrain to SENSR from LP
    'adult_expt108': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt100"),
        'trainer_kwargs': {
            'epsilon': 7.5e-2, # was 3e-2
            'alpha': 0.0175, #was 0.02
        },
    },

    # Retrain to LP from SENSR
    'adult_expt109': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt101"),
        'trainer_kwargs': {
            'epsilon': 0.25,
            'alpha': 0.875, # was 0.95
        }
    },

    # F-PGD training (SENSR)
    'adult_expt110': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'fairpgdtrainer',
        'metric': 'SENSR',
        'dataset': 'adult',
        'trainer_kwargs': {
            'epsilon': 1e-2,
            'alpha': 0.4,
        }
    },
    # Previously Missing config! 
    # Retrain to PGD from LP
    'adult_expt1101': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt110"),
        'trainer_kwargs': {
            'epsilon': 0.25,
            'alpha': 0.05,
        }
    },
    # Previously Missing config! 
    # Retrain to PGD from SENSR
    'adult_expt1102': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'data_frac': SMALLEST_DATA_FRAC,
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt103"),
        'trainer_kwargs': {
            'epsilon': 3e-2,
            'alpha': 0.05,
        }
    },

    ###################
    # Covar shift
    ###################

    # Standard -> F-IBP-LP (shifted)
    'adult_expt111': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.1, # was 0.2
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt0"),
        'max_steps': 5000
    },

    # R-IBP -> F-IBP-LP (shifted)
    'adult_expt112': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.2,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt102"),
        'max_steps': 5000,
    },

    # F-IBP-LP -> F-IBP-LP (shifted)
    'adult_expt113': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.5, # was 0.2
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt100"),
        'max_steps': 5000,
    },

    ###################
    # label shift
    ###################

    # Standard -> F-IBP-LP (shifted)
    'adult_expt114': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.3, # 0.0 (99, 0.99) 0.1 (95.9, 0.96) 0.3 (96, 0.24) 0.5 (95.5, 0.40)
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt0"),
        'max_steps': 5000
    },

    # R-IBP -> F-IBP-LP (shifted)
    'adult_expt115': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.4, # 0.1 (95.8, 0.33) 0.2 (95.9, 0.323) 0.4 (94.8, 0.06) 
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt102"),
        'max_steps': 5000
    },

    # F-IBP-LP -> F-IBP-LP (shifted)
    'adult_expt116': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.45, # 0.1 (95.6, 0.922) 0.3 (89, 0.81) 0.5 (52, 0.03) 0.4 (95.9, 0.081) 
            # I had to change this to get a good result
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt100"),
        'max_steps': 5000 # This is the most fickle result ever haha
    },

    #############
    # data fraction experiments with covar shift
    #############
    'adult_datafrac_covshift_template': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.35,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt100"),
    },
    
     # Training at different scales
    'adult_modelscale_template': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.075,
        },
        'model_kwargs': {
            'hidden_lay': 1,
            'hidden_dim': 128,
        },
    },
    
    'adult_expt119': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.1, # 0.1, 0.5 or 0.2 or 0.15 (trivial accuracy)
        },
    },
    'adult_expt120': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.5,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt102"),
    },

    'adult_expt121': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt0"),
        'trainer_kwargs': {
            'epsilon': 3.5e-2,
            'alpha': 0.0025, # 0.005 (59.7, 0.84) 0.01 (51.09, 0.99) 0.0025 (61.4, 0.78)
        }
    },
    'adult_expt122': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'adult',
        'trainer_kwargs': {
            'epsilon': 3.5e-2,
            'alpha': 0.005, # 0.005 (58.9, 0.42)
        }
    },
    'adult_expt123': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'SENSR',
        'dataset': 'adult',
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt102"),
        'trainer_kwargs': {
            'epsilon': 3.5e-2,
            'alpha': 0.005, #0.025 (alm. trivial accuracy) 0.01 (51.1, 0.998) 0.005 (58.9, 0.39)
        }
    },
    
    
    # Retrain to LP from SENSR
    'adult_expt125': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'unlabeled',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.94,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt102"),
        'max_steps': 5000,
    },
    
    # Retrain to LP from SENSR
    'adult_expt127': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'dataless',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 1.0,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.97,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt102"),
        'max_steps': 5000,
    },
    
    'adult_expt128': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'robust',
        'metric': 'LP',
        'dataset': 'adult',
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
    
    'adult_expt129': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'num_adapt': 1,
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'shift': DEFAULT_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.05,
            'alpha': 0.0025,
        },
        'model_kwargs': {
            'hidden_dim': 128,
            'hidden_lay': 2,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt128"),
        'max_steps': 15000,
        'use_weighted_ce': True,
    },

    'adult_expt130': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'trainer_kwargs': {
            #'epsilon': 0.1,
            'epsilon': 0.25,
            'alpha': 0.05,
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
    },
    
    'adult_expt131': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'certifier',
        'metric': 'LP',
        'dataset': 'adult',
        'num_adapt': 1,
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'dataset_kwargs': {'label_shift': DEFAULT_LABEL_SHIFT_FACTOR},
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.4, # 0.1 (95.8, 0.33) 0.2 (95.9, 0.323) 0.4 (94.8, 0.06) 
        },
        'model_kwargs': {
            'hidden_dim': 32,
            'hidden_lay': 2,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt130"),
    },
    
    'adult_expt132': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'unlabeled',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 2*SMALLEST_DATA_FRAC,
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.94,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt101"),
        'max_steps': 5000,
    },
    
    # Retrain to LP from SENSR
    'adult_expt133': {
        'out_dir': OUT_DIR,
        'model': 'ffn',
        'trainer': 'dataless',
        'metric': 'LP',
        'dataset': 'adult',
        'data_frac': 1.0,
        'trainer_kwargs': {
            'epsilon': 0.1,
            'alpha': 0.97,
        },
        'model_ckpt': get_best_ckpt(f"{OUT_DIR}/adult_expt101"),
        'max_steps': 5000,
    },    
    
}



for data_frac in DATA_FRACS:
    _d = adult_expts['adult_datafrac_covshift_template'].copy()
    _d['data_frac'] = data_frac
    adult_expts[f'adult_expt117_{data_frac:0.3f}'] = _d

WIDTHS = [8, 12, 16, 24, 64, 128, 256, 512]
DEPTHS = [1, 2, 3]
import copy
for depth in DEPTHS:
    for width in WIDTHS:
        _d = copy.deepcopy(adult_expts['adult_modelscale_template'])
        _d['model_kwargs']['hidden_dim'] = width
        _d['model_kwargs']['hidden_lay'] = depth
        adult_expts[f'adult_expt118_%s_%s'%(depth, width)] = _d
        del(_d)
        

    