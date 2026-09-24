import numpy as np


class LSTMInference:
    def __init__(self, model_path):
        weights = np.load(model_path)

        self.hidden_size = int(weights["hidden_size"])

        self.w_f = weights["w_f"]
        self.b_f = weights["b_f"]
        self.w_i = weights["w_i"]
        self.b_i = weights["b_i"]
        self.w_cm = weights["w_cm"]
        self.b_cm = weights["b_cm"]
        self.w_o = weights["w_o"]
        self.b_o = weights["b_o"]
        self.w_out = weights["w_out"]
        self.b_out = weights["b_out"]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)

        batch_size = x.shape[0]
        h_t = np.zeros((batch_size, self.hidden_size))
        c_t = np.zeros((batch_size, self.hidden_size))

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            combined = np.concatenate((x_t, h_t), axis=1)

            forget = self.sigmoid(combined @ self.w_f + self.b_f)
            input_gate = self.sigmoid(combined @ self.w_i + self.b_i)
            candidate = np.tanh(combined @ self.w_cm + self.b_cm)
            output = self.sigmoid(combined @ self.w_o + self.b_o)

            c_t = c_t * forget + input_gate * candidate
            h_t = np.tanh(c_t) * output

        probability = self.sigmoid(
            h_t @ self.w_out + self.b_out
        ).reshape(-1)

        direction = (probability >= 0.5).astype(int)

        return float(probability[0]), int(direction[0])