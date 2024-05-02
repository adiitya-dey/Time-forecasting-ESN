from darts.datasets import AirPassengersDataset, AusBeerDataset




class Dataset:

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
            

            

