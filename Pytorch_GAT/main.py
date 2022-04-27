from train import train, evaluate

if __name__ == '__main__':
    GATmodel = train(epoch=100)
    evaluate(GATmodel)
