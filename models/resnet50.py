from torchvision import datasets
import torchvision.transforms as transforms


# create transforms and Normalization 
my_transforms = transforms.Compose([transforms.Resize(size=(244,244)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225] )])

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(val_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)

loaders_scratch = {
    'train':train_loader,
    'valid':val_loader,
    'test':test_loader,
}