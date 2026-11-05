#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np


def read_params(filename="data/in.dat"):
    """Чтение размеров матрицы."""
    with open(filename, "r") as f:
        N = int(f.readline().strip())
        M = int(f.readline().strip())

    return N, M


def auxiliary_arrays_determination(M, numprocs):
    """Расчет количества строк и смещений для каждого процесса."""
    base, rem = divmod(M, numprocs)

    rcounts = np.array(
        [base + (1 if i < rem else 0) for i in range(numprocs)],
        dtype=np.int32
    )

    displs = np.zeros(numprocs, dtype=np.int32)
    if numprocs > 1:
        displs[1:] = np.cumsum(rcounts[:-1])

    return rcounts, displs


def main():
    # Инициализация MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    # Чтение параметров на процессе 0
    if rank == 0:
        N_val, M_val = read_params("data/in.dat")
        params = np.array([N_val, M_val], dtype=np.int32)
    else:
        params = np.empty(2, dtype=np.int32)

    # Рассылка параметров всем процессам
    comm.Bcast([params, MPI.INT], root=0)
    N, M = int(params[0]), int(params[1])

    # Расчет разбиения матрицы на всех процессах
    rcounts, displs = auxiliary_arrays_determination(M, numprocs)

    local_m = int(rcounts[rank])
    local_elements = local_m * N

    # Подготовка буферов
    A_part = np.empty((local_m, N), dtype=np.float64)
    x = np.empty(N, dtype=np.float64)
    b_part = np.empty(local_m, dtype=np.float64)

    # Синхронизация перед началом измерения
    comm.Barrier()
    start_time = MPI.Wtime()

    # Чтение матрицы и распределение через Scatterv
    if rank == 0:
        A = np.empty((M, N), dtype=np.float64)

        with open("data/AData.dat", "r") as f:
            A = np.fromstring(f.read(), sep=" ", dtype=np.float64)

        A = A.reshape(M, N)

        sendcounts_A = rcounts * N
        displs_A = displs * N

        sendbuf_A = [
            A.ravel(),
            sendcounts_A,
            displs_A,
            MPI.DOUBLE
        ]
    else:
        A = None
        sendbuf_A = None

    comm.Scatterv(
        sendbuf_A,
        [A_part.ravel(), local_elements, MPI.DOUBLE],
        root=0
    )

    # Чтение и рассылка вектора x
    if rank == 0:
        with open("data/xData.dat", "r") as f:
            x = np.fromstring(f.read(), sep=" ", dtype=np.float64)

    comm.Bcast([x, MPI.DOUBLE], root=0)

    # Локальное вычисление b_part = A_part * x
    np.dot(A_part, x, out=b_part)

    # Сбор результатов через Gatherv
    if rank == 0:
        b = np.empty(M, dtype=np.float64)
        recvbuf_b = [b, rcounts, displs, MPI.DOUBLE]
    else:
        b = None
        recvbuf_b = None

    comm.Gatherv(
        [b_part, local_m, MPI.DOUBLE],
        recvbuf_b,
        root=0
    )

    # Завершение измерения после сбора результатов
    comm.Barrier()
    elapsed_time = MPI.Wtime() - start_time

    # Запись результата и статистики эксперимента
    if rank == 0:
        with open("results/Results_parallel_scatterv.dat", "w") as f:
            for value in b:
                f.write(f"{value}\n")

        print(f"N: {N}, M: {M}")
        print(f"Execution time: {elapsed_time:.6f} seconds")

        with open("results/timings_1_2.csv", "a") as f:
            f.write(
                f"{numprocs},{N},{M},{elapsed_time:.6f}\n"
            )


if __name__ == "__main__":
    main()