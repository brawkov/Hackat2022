import numpy as np
import pandas as pd


def read_signals(filename):
    samples_count = 5000

    c = ['name', 'x', 'y']
    for i in range(0, samples_count):
        c.append(f'v{i}')
    c = c + ['cluster', 'p0', 'p1', 'p2', 'p3']

    df = pd.read_csv(filename, names=c, dtype=np.float32)
    df = df.set_index('name', drop=True)

    return df


def write_signals(df, filename):
    df.to_csv(filename, header=False)


if __name__ == "__main__":
    # Example
    df = read_signals('signals.csv')
    print(df)
    write_signals(df, 'signals-out.csv')