#!/usr/bin/env python3
"""
Генерация тестовых данных для ЛР №4
"""
import numpy as np
import os

def generate_test_data(M, N, seed, output_dir):
    """Генерация тестовых данных"""
    np.random.seed(seed)
    
    b = np.random.rand(M)
    x = np.random.rand(N)
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.savetxt(os.path.join(output_dir, "bData.dat"), b)
    
    with open(os.path.join(output_dir, "in.dat"), "w") as f:
        f.write(f"{N}\n{M}\n")

if __name__ == "__main__":
    generate_test_data(20000, 1000, 42, "data")