
import argparse
import torch
import torch.optim as optim
from model import TMC
from data_loader import MultiViewData
import warnings
from torch.autograd import Variable
warnings.filterwarnings("ignore")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer():
    def __init__(self):
        """
        Initializes the trainer with hyperparameters like batch size, epochs, lambda for KL annealing, and learning rate.
        Sets up the dataset and DataLoader for both training and testing phases.
        Initializes the TMC model and the Adam optimizer.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                            help='input batch size for training [default: 100]')
        parser.add_argument('--epochs', type=int, default=500, metavar='N',
                            help='number of epochs to train [default: 500]')
        parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                            help='gradually increase the value of lambda from 0 to 1')
        parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                            help='learning rate')
        self.args = parser.parse_args()
        self.args.data_name = 'handwritten_6views'
        self.args.data_path = 'datasets/' + self.args.data_name
        self.args.dims = [[240], [76], [216], [47], [64], [6]]
        self.args.views = len(self.args.dims)

        self.train_loader = torch.utils.data.DataLoader(
            MultiViewData(self.args.data_path, train=True), batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            MultiViewData(self.args.data_path, train=False), batch_size=self.args.batch_size, shuffle=False)
        N_mini_batches = len(self.train_loader)
        print('The number of training images = %d' % N_mini_batches)

        self.model = TMC(10, self.args.views, self.args.dims, self.args.lambda_epochs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        self.model.cuda()

    def train(self, epoch):
        """
        Trains the TMC model on the data
        """
        # Sets the model to training mode.
        self.model.train()
        
        # loss for each iteration is tracked using an AverageMeter instance
        loss_meter = AverageMeter()
        
        # Iterates over batches of data, converting them to variables and moving them to the GPU.
        for batch_idx, (data, target) in enumerate(self.train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.long().cuda())
        
            # refresh the optimizer
            self.optimizer.zero_grad()
            (self.optimizer.step()).evidences, evidence_a, loss = self.model(data, target, epoch)
        
            # The gradients are computed via backpropagation 
            loss.backward()
        
            # Optimizer updates the model parameters 
            self.optimizer.step()
            loss_meter.update(loss.item())
            
    def test(self, epoch):
        """
        Returns the average loss and accuracy for the test set.
        """
        
        # Sets the model to evaluation mode to disable dropout or batch normalization effects during inference.
        self.model.eval()

        # loss for each iteration is tracked using an AverageMeter instance
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            
            # Gradient computation disabled
            with torch.no_grad():
                target = Variable(target.long().cuda())
                evidences, evidence_a, loss = self.model(data, target, epoch)
                _, predicted = torch.max(evidence_a.data, 1)
                
                # Predictions are made, and the number of correct predictions is counted to calculate the accuracy.
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())
        return loss_meter.avg, correct_num/data_num

    def run(self):
        """
        Perform the training process for the specified number of epochs.
        """

        for epoch in range(1, self.args.epochs + 1):
            self.train(epoch)

        # model is evaluated on the test set.
        test_loss, acc = self.test(epoch)
        return test_loss, acc