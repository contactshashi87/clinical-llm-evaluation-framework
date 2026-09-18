from .reporting import ReportManager
from .defect_manager import generate_defects
import pandas as pd
from .settings import OUTPUT_DIR

def read(name):
    p=OUTPUT_DIR/f'{name}.csv'
    return pd.read_csv(p) if p.exists() else None

def main():
    rm=ReportManager(); e=read('evaluation_results'); s=read('safety_results'); r=read('regression_results'); d=generate_defects(e,s); rm.generate_benchmark(e,s,r); print(f'Defects: {len(d)}'); print(rm.export_excel())
if __name__=='__main__': main()
