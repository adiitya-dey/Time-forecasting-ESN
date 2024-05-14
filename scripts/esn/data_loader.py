from darts.datasets import AirPassengersDataset, AusBeerDataset
import numpy as np



class DartsDataset:

    def __call__(self, data):
        self.data = data
        match self.data.lower():
            case "airpassengers":
                return {"name": "AirPassengers",
                        "dataset": AirPassengersDataset(),
                        "input": 1,
                        "output": 1}
            case "ausbeer":
                return {"name": "AirPassengers",
                        "dataset": AusBeerDataset(),
                        "input": 1,
                        "output": 1}
            

            
class AugmentedDataset:

    def __init__(self):
        self.points = np.linspace(1,50, 500)
        self.noise = np.random.randn(len(self.points))

    def __call__(self, data):
        self.data = data
        match self.data.lower():
            case "linear":
               return self.points
            case "seasonal":
                return self.seasonal(self.points)
            case "noise":
                return self.noise
            case "linear seasonal":
                return self.linear_seasonal(self.points)
            case "linear noise":
                return self.points + self.noise
            case "seasonal noise":
                return self.seasonal(self.points) + self.noise
            case "linear seasonal noise":
                return self.linear_seasonal(self.points) + self.noise

            


    def seasonal(self,x):
        return np.sin(2 * np.pi * 10 * x)
    
    def linear_seasonal(self,x, frequency=10, amplitude=1, trend_slope=0.1):
        
        trend = trend_slope * x
        oscillations = amplitude * np.cos(2 * np.pi * frequency * x)
        y = trend + oscillations
        return y
