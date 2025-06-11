import pandas as pd
import numpy as np

df = pd.read_excel("mqtt_data.xlsx")
sv = df["SV"].values
pv = df["PV"].values

# Tracking error
tracking_error = sv - pv
dt = 10

de = np.diff(tracking_error) / dt
dde = np.diff(de) / dt

e = np.column_stack([
    tracking_error[2:],  # e
    de[1:]               # de
])
de_full = np.column_stack([
    de[1:],              # de
    dde                  # dde
])

np.savez("training_data_uav.npz", e=e, de=de_full)
print("Saved as training_data_uav.npz (with shape e:(N,2), de:(N,2))")
