import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Visualization:
    def __init__(self, data):
        self.data = data

    def visualization_energy(self):
        plt.figure(figsize=(10, 12))

        plt.plot(self.data.index, self.data["EnergyProduced"])
        plt.xlabel("Date")
        plt.ylabel("Energy Production")
        plt.title("Energy Production Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualization_every_column(self):
        plt.figure(figsize=(10, 12))

        plt.subplot(5, 1, 1)
        plt.plot(self.data.index, self.data["EnergyProduced"])
        plt.xlabel("Date")
        plt.ylabel("Energy Production")
        plt.title("Energy Production Over Time")
        plt.grid(True)

        plt.subplot(5, 1, 2)
        plt.plot(self.data.index, self.data["AverageLufttemperatur"], label="Average Temperature")
        plt.xlabel("Date")
        plt.ylabel("Average Temperature")
        plt.title("Average Temperature Over Time")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, 1, 3)
        plt.plot(self.data.index, self.data["AverageNederbördsmängd"], label="Precipitation")
        plt.xlabel("Date")
        plt.ylabel("Precipitation")
        plt.title("Precipitation Over Time")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, 1, 4)
        plt.plot(self.data.index, self.data["AverageSnödjup"], label="Snow Depth")
        plt.xlabel("Date")
        plt.ylabel("Snow Depth")
        plt.title("Snow Depth Over Time")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, 1, 5)
        plt.plot(self.data.index, self.data["AverageSolskenstid"], label="Sunshine Duration")
        plt.xlabel("Date")
        plt.ylabel("Sunshine Duration")
        plt.title("Sunshine Duration Over Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def normalize(self):
        return pd.DataFrame(MinMaxScaler().fit_transform(self.data), columns=self.data.columns, index=self.data.index)
    
    def visualization_with_weather(self):
        normalize_data = self.normalize()
        plt.figure(figsize=(10, 6))
        plt.plot(normalize_data.index, normalize_data["EnergyProduced"], label="Energy Produced")
        plt.plot(normalize_data.index, normalize_data["AverageLufttemperatur"], label="Average Temperature")
        plt.plot(normalize_data.index, normalize_data["AverageNederbördsmängd"], label="Precipitation")
        plt.plot(normalize_data.index, normalize_data["AverageSnödjup"], label="Snow Depth")
        plt.plot(normalize_data.index, normalize_data["AverageSolskenstid"], label="Sunshine Duration")
        plt.xlabel("Date")
        plt.ylabel("Energy Production")
        plt.title("Energy Production Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualization_with_radiation(self):
        plt.figure(figsize=(10, 12))

        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data["EnergyProduced"])
        plt.xlabel("Date")
        plt.ylabel("Energy Production")
        plt.title("Energy Production Over Time")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.data.index, self.data["UV-irradiation"], label="UV irradiation")
        plt.xlabel("Date")
        plt.ylabel("UV irradiation")
        plt.title("UV irradiation Over Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def visualization_radiation(self):
        plt.figure(figsize=(10, 12))

        plt.plot(self.data.index, self.data["UV-irradiation"], label="UV irradiation")
        plt.xlabel("Date")
        plt.ylabel("UV irradiation")
        plt.title("UV irradiation Over Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()