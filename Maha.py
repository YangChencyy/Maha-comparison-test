import lib_generation
from dataset import *
from models.models import MNIST_Net, Fashion_MNIST_Net, Cifar_10_Net
from OOD_Generate_Mahalanobis import Generate_Maha
from OOD_Regression_Mahalanobis import Regression_Maha

import argparse

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dic = {
    'MNIST': MNIST_dataset,
    'FashionMNIST': Fashion_MNIST_dataset, 
    'Cifar_10': Cifar_10_dataset,
    'SVHN': SVHN_dataset, 
    'Imagenet_r': TinyImagenet_r_dataset,
    'Imagenet_c': TinyImagenet_c_dataset
}


data_model = {
    'MNIST': MNIST_Net,
    'FashionMNIST': Fashion_MNIST_Net, 
    'Cifar_10': Cifar_10_Net   
}

def main():

    parser = argparse.ArgumentParser(description="Mahalanobis parameters")

    # Add a positional argument for the number
    parser.add_argument("InD_Dataset", type=str, help="The name of the InD dataset.")
    parser.add_argument("train_batch_size", type=int, help="train_batch_size")
    parser.add_argument("test_batch_size", type=int, help="test_batch_size")
    # parser.add_argument("gpu", type=int, help="number of gpu")

    # Parse the command-line arguments
    args = parser.parse_args()


    train_set, test_set, trloader, tsloader = data_dic[args.InD_Dataset](batch_size = args.train_batch_size, 
                                                                    test_batch_size = args.test_batch_size)
    OOD_sets, OOD_loaders = [], []
    parent_dir = os.getcwd()
    if args.InD_Dataset == 'Cifar_10':
        OOD_Dataset = ['SVHN', 'Imagenet_r', 'Imagenet_c']
        net_name = "densenet"
        net_Maha = torch.load('./pre_trained/' + net_name + '_' + args.InD_Dataset + '.pth', map_location = "cuda:0")

        # Get all OOD datasets     
        for dataset in OOD_Dataset:
            _, OOD_set, _, OODloader = data_dic[dataset](batch_size = args.train_batch_size, 
                                                        test_batch_size = args.test_batch_size)
            OOD_sets.append(OOD_set)
            OOD_loaders.append(OODloader)

    else:
        if args.InD_Dataset == 'MNIST':
            # OOD_Dataset = ['Imagenet_r']
            OOD_Dataset = ['FashionMNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']
        elif args.InD_Dataset == 'FashionMNIST':
            OOD_Dataset = ['MNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']

        net_name = 'dnn'
        net_Maha = data_model[args.InD_Dataset]()
        net_Maha.load_state_dict(torch.load(os.path.join(parent_dir, 'pre_trained/' + args.InD_Dataset + "_net.pt")))

        # Get all OOD datasets     
        for dataset in OOD_Dataset:
            _, OOD_set, _, OODloader = data_dic[dataset](batch_size = args.train_batch_size, 
                                                        test_batch_size = args.test_batch_size, into_grey = True)
            OOD_sets.append(OOD_set)
            OOD_loaders.append(OODloader)

        # net_Maha = data_model[InD_Dataset]()
    
    outf = parent_dir + '/output/' + net_name  + '/'
    if os.path.isdir(outf) == False:
        os.mkdir(outf)

    
    Generate_Maha(net_Maha, outf, args.InD_Dataset, OOD_Dataset, trloader, tsloader, 
                OOD_loaders, net_name, num_classes = 10)
    Regression_Maha(args.InD_Dataset, OOD_Dataset, net_name, outf)


if __name__ == "__main__":
    main()