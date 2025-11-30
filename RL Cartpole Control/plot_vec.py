import matplotlib.pyplot as plt
import numpy as np

# path to the stored returns
train = np.loadtxt('logs/state_pixels/train_avg_returns.txt')
evals = np.loadtxt('logs/state_pixels/model_aac.log')

plt.figure(figsize=(10,6))
plt.plot(train, label="Training Average Return")
plt.plot(evals, label="Evaluation Average Return")

plt.xlabel("Epoch")
plt.ylabel("Average Return")
plt.title("Training vs Evaluation Returns (200 Epochs)")
plt.legend()
plt.grid(True)

plt.savefig("training_vs_evaluation_returns.png", dpi=300)
plt.show()
