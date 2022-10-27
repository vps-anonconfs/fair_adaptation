from .german import GermanDataset
from .adult import AdultDataset
from .credit import CreditDataset


def get_dataset(name, **dataset_kwargs):
    name = name.lower()
    if name == 'german':
        return GermanDataset(**dataset_kwargs)
    elif name == 'adult':
        return AdultDataset(**dataset_kwargs)
    elif name == 'credit':
        return CreditDataset(**dataset_kwargs)