from models import FairnessCertifier

def get_default_model(dataset, num_redacted_features=0):
    """
    Returns the default model architecture for the dataset
    :param dataset: Name of the dataset
    :param num_redacted_features: Number of redacted features that may have been removed later (eg. sensitive)
    """
    dataset = dataset.upper()
    assert dataset in ["ADULT", "CREDIT", "GERMAN", "CRIME", "STUDENT", "AMEX"]
    
    if(dataset=="ADULT"):
        #self.class_weights = torch.Tensor([1,2*3.017])
        class_weights = torch.Tensor([1,1])
        in_dim = 63 
        num_cls = 2
    elif(dataset=="CREDIT"):
        #self.class_weights = torch.Tensor([1,3.508])
        class_weights = torch.Tensor([1,1])
        in_dim = 146
        num_cls = 2
    elif(dataset=="GERMAN"):
        class_weights = torch.Tensor([2.319, 1])
        #self.class_weights = torch.Tensor([1,1])
        in_dim = 62
        num_cls = 2
    elif(dataset=="CRIME"):
        #self.class_weights = torch.Tensor([1,2*42.081])
        class_weights = torch.Tensor([1,1])
        in_dim = 54
        num_cls = 2
    elif(dataset=="STUDENT"):
        #self.class_weights = torch.Tensor([1,2*42.081])
        class_weights = torch.Tensor([1,1])
        in_dim = 25
        num_cls = 2
    elif(dataset=="AMEX"):
        #self.class_weights = torch.Tensor([1,2*42.081])
        class_weights = torch.Tensor([1,1])
        in_dim = 155
        num_cls = 2
        
    in_dim -= num_redacted_features
    model = FullConnected.FullConnected(input_dimension=in_dim, num_cls=num_cls, hidden_dim=128, hidden_lay=1)
    model.class_weights = class_weights
    return model