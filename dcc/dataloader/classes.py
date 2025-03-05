import pandas as pd
import os
import numpy as np

cellmap_classes_csv = os.path.join(os.path.dirname(__file__), "cellmap_clases.csv")

def get_groupping(target_classes):
    df = pd.read_csv(cellmap_classes_csv)
    result = df[df['field_name'].isin(target_classes)]
    groupping = []
    for index, row in result.iterrows():
        field_name = row['field_name']
        class_id = row['class_id']
        groupping.append((field_name, [class_id]))
    groupping.sort(key=lambda x: target_classes.index(x[0]))
    return groupping
