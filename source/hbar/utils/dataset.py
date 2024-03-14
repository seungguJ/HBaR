from .. import *


def get_dataset_from_code(code, batch_size):
    """ interface to get function object
    Args:
        code(str): specific data type
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    dataset_root = "/workspace/data"
    if code == 'cifar100':
        train_loader, test_loader = get_cifar100_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'CIFAR100'))
    elif code == 'cifar10':
        train_loader, test_loader = get_cifar10_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'CIFAR10'))
    elif code == 'svhn':
        train_loader, test_loader = get_svhn_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'SVHN'))
    else:
        raise ValueError("Unknown data type : [{}] Impulse Exists".format(code))

    return train_loader, test_loader


def get_cifar10_data(data_folder_path, batch_size=64):
    """ cifar10 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
    ])

    train_data = datasets.CIFAR10(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR10(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def get_svhn_data(data_folder_path, batch_size=64):
    """ svhn data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        transforms.ToTensor(),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.SVHN(data_folder_path, split='train', 
        download=True, transform=transform_train)
    test_data  = datasets.SVHN(data_folder_path, split='test', 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def get_cifar100_data(data_folder_path, batch_size=64):
    """ cifar100 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.CIFAR100(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR100(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)
    print(f'cifar100')
    return train_loader, test_loader