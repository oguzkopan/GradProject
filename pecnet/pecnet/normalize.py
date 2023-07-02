import numpy as np 

class WindowNormalizer():
    def fit(self, data):
        self.mean = np.mean(data, 1)
    def fit_transform(self, data):
        self.mean = np.mean(data, 1)
        return data - self.mean[:, np.newaxis]
    def transform(self, data, is_window=True):
        if self.mean is None:
            raise ValueError("Fit Before Transforming")
        norm = data - self.mean  

        return norm[:, np.newaxis] if is_window else norm