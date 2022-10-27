model_defaults = {
    'ffn': {
        'model_kwargs': {
            'hidden_lay': 1,
            'hidden_dim': 128,
        },
        'trainer_kwargs': {
            'epsilon': 5e-2,
            'optimizer': 'adam',
            'learning_rate': 1e-3,
            'alpha': 0.5,
        }
    }
}
