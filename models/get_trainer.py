from .certifier_trainer import FairnessCertifier
from .unlab_certifier_trainer import UnlabFairnessCertifier
from .robust_trainer import RobustClassifier
from .constant_trainer import ConstantClassifier
from .fairpgd_trainer import PGDFairnessTrainer
from .unlabeled_adapter import UnlabeledAdapter

def get_trainer(base_model, model_name, **alg_kwargs):
    if model_name == 'certifier':
        return FairnessCertifier(base_model, **alg_kwargs)
    elif model_name == 'unlab_certifier':
        return UnlabFairnessCertifier(base_model, **alg_kwargs)
    elif model_name == 'unlabeled':
        return UnlabeledAdapter(base_model, **alg_kwargs)
    elif model_name == 'dataless':
        return UnlabeledAdapter(base_model, **alg_kwargs)
    elif model_name == 'robust':
        return RobustClassifier(base_model, **alg_kwargs)
    elif model_name == 'constant':
        return ConstantClassifier(base_model, **alg_kwargs)
    elif model_name == 'fairpgdtrainer':
        return PGDFairnessTrainer(base_model, **alg_kwargs)
    raise AssertionError(f'{model_name} unsupported.')
