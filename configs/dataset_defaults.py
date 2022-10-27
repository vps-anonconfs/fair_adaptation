dataset_defaults = {
    'german': {
        'dataset_kwargs': {
            'path': 'cache/german.csv',
            'sensitive_features': ['status_sex_A91', 'status_sex_A93', 'status_sex_A94', 'status_sex_A92'],
            'drop_columns': [],
            'use_sens': False
        },
        'model': 'ffn',
        'model_kwargs': {'input_dimension': 58},
        'max_steps': 5000,
        'use_weighted_ce': True
    },
    'credit': {
        'dataset_kwargs': {
            'path': 'cache/credit.csv',
            'sensitive_features': ['x2_1.0', 'x2_2.0'],
            'drop_columns': [],
            'use_sens': False
        },
        'model': 'ffn',
        'model_kwargs': {'input_dimension': 144},
        'max_steps': 5000,
        'use_weighted_ce': True
    },
    'adult': {
        'dataset_kwargs': {
            'path': ['cache/adult.data.csv', 'cache/adult.test.csv'],
            'sensitive_features': ['sex_Male', 'sex_Female'],
            'drop_columns': ['native-country'],
            'use_sens': False
        },
        'model': 'ffn',
        'model_kwargs': {'input_dimension': 61},
        'max_steps': 10000,
        'use_weighted_ce': True
    },
    'student': {
        'dataset_kwargs': {
            'path': 'cache/studentInfo.csv',
            'sensitive_features': ['gender_M', 'gender_F'],
            'drop_columns': ['code_module', 'code_presentation', 'region'],
            'use_sens': False
        },
        'model': 'ffn',
        'model_kwargs': {'input_dimension': 21},
        'max_steps': 1500,
    }
}
