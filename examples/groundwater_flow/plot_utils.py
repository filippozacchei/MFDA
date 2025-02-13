import numpy as np
import matplotlib.pyplot as plt

def plot_results(X_test, y_test, model):
    x_data = y_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    X,Y = np.meshgrid(x_data, y_data)

    samples = [ 1, 10, 100]
    #Plot POD coefficients: LF vs HF
    fig = plt.figure(figsize=(12,12))
    plt.subplots_adjust(hspace=0.5)
    title = 'True vs Reconstucted signals with NN'
    fig.suptitle(title, fontsize=14)

    for mode in range(3):
        ax = fig.add_subplot(331 + mode)
        pcm = plt.pcolormesh(X, Y, y_test[mode, :].reshape((5, 5)))
        ax.title.set_text('True signal sample: ' + str(samples[mode]))
        plt.colorbar(pcm, ax=ax)
        
        ax = fig.add_subplot(331 + mode + 3)
        reconstructed_sample = np.array(model(X_test[mode, :].reshape((1, 64)))).reshape((5, 5))
        err = y_test[mode, :].reshape(5, 5) - reconstructed_sample
        pcm = plt.pcolormesh(X, Y, reconstructed_sample.reshape((5, 5)))
        ax.title.set_text('NN prediction sample: ' + str(samples[mode]))
        plt.colorbar(pcm, ax=ax)
        
        ax = fig.add_subplot(331 + mode + 6)
        pcm = plt.pcolormesh(X, Y, err)
        ax.title.set_text('Reconst. error sample: ' + str(samples[mode]))
        plt.colorbar(pcm, ax=ax)
    
    plt.show()