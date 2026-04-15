import pandas as pd
from pathlib import Path
s=pd.read_csv(next(Path("/").rglob("shallow_4hz_results.csv")))
d=pd.read_csv(next(Path("/").rglob("deep_0hz_results.csv")))
out=pd.DataFrame([
["Deep",0,round((1-d["best_test_misclass"]).mean()*100,2),round((1-d["last_test_misclass"]).mean()*100,2),92.3,92.5],
["Shallow",4,round((1-s["best_test_misclass"]).mean()*100,2),round((1-s["last_test_misclass"]).mean()*100,2),94.6,93.9]],
columns=["Model","Lowpass [Hz]","Peak test accuracy during training [%]","Test accuracy at training stop [%]","GitHub reproduction [%]","Original paper [%]"])
out.to_csv("comparison_table.csv",index=False); print(out.to_string(index=False))