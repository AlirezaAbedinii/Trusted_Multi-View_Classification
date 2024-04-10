import os
from train import Trainer
from test_cuda import check_cuda

if __name__ == "__main__":
    cuda_available, cuda_device_id = check_cuda()
    if cuda_available:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    else:
        raise Exception("CUDA is not available. Try Installing it first")

    trainer = Trainer()
    test_loss, acc = trainer.run()
    print('######## acc: {:.4f}'.format(acc))
    print('######## loss: {:.4f}'.format(test_loss))