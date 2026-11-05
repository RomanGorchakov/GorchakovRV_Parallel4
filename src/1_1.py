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
    base = M // numprocs
    rem = M % numprocs

    rcounts = np.array(
        [base + (1 if i < rem else 0) for i in range(numprocs)],
        dtype=np.int32
    )

    displs = np.zeros(numprocs, dtype=np.int32)
    if numprocs > 1:
        displs[1:] = np.cumsum(rcounts[:-1])

    return rcounts, displs


def read_matrix_part(f, local_m, N):
    """Чтение части матрицы из файла."""
    A_part = np.empty((local_m, N), dtype=np.float64)

    values = []

    while len(values) < local_m * N:
        line = f.readline()

        if not line:
            raise ValueError("Недостаточно данных в файле матрицы")

        values.extend(map(float, line.split()))

    A_part[:] = np.array(
        values[:local_m * N],
        dtype=np.float64
    ).reshape(local_m, N)

    return A_part


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    # Чтение параметров задачи
    if rank == 0:
        N, M = read_params("data/in.dat")
        params = np.array([N, M], dtype=np.int32)
    else:
        params = np.empty(2, dtype=np.int32)

    comm.Bcast([params, MPI.INT], root=0)
    N, M = int(params[0]), int(params[1])

    rcounts, displs = auxiliary_arrays_determination(M, numprocs)
    local_m = int(rcounts[rank])

    # Синхронизация перед измерением времени
    comm.Barrier()
    start_time = MPI.Wtime()

    # Распределение матрицы A с использованием Send/Recv
        # Распределение матрицы A с использованием Send/Recv
    if rank == 0:
        with open("data/AData.dat", "r") as f:
            A = np.fromstring(f.read(), sep=" ", dtype=np.float64)

        A = A.reshape(M, N)

        # Процесс 0 также участвует в вычислениях
        A_part = A[displs[0]:displs[0] + local_m].copy()

        # Отправка частей матрицы остальным процессам
        for k in range(1, numprocs):
            rows = int(rcounts[k])
            offset = int(displs[k])

            A_send = A[offset:offset + rows].copy()
            comm.Send([A_send, MPI.DOUBLE], dest=k, tag=0)

    else:
        A_part = np.empty((local_m, N), dtype=np.float64)
        comm.Recv([A_part, MPI.DOUBLE], source=0, tag=0)

    # Рассылка вектора x
    x = np.empty(N, dtype=np.float64)

    if rank == 0:
        with open("data/xData.dat", "r") as f:
            for i in range(N):
                x[i] = float(f.readline())

    comm.Bcast([x, MPI.DOUBLE], root=0)

    # Локальное вычисление b_part = A_part * x
    b_part = np.dot(A_part, x)

    # Сбор результатов на процессе 0
    if rank == 0:
        b = np.empty(M, dtype=np.float64)

        # Копирование результата процесса 0
        b[displs[0]:displs[0] + local_m] = b_part

        for k in range(1, numprocs):
            rows = int(rcounts[k])
            offset = int(displs[k])

            comm.Recv(
                [b[offset:offset + rows], MPI.DOUBLE],
                source=k,
                tag=1
            )

        elapsed_time = MPI.Wtime() - start_time

        # Сохранение вектора результата
        with open("results/Results_parallel_send_recv.dat", "w") as f:
            for value in b:
                f.write(f"{value}\n")

        # Вывод времени для анализа масштабируемости
        print(f"N: {N}, M: {M}")
        print(f"Execution Time: {elapsed_time} seconds")

        # Сохранение параметров эксперимента
        with open("results/timings_1_1.csv", "a") as f:
            f.write(
                f"{numprocs},{N},{M},{elapsed_time}\n"
            )

    else:
        comm.Send([b_part, MPI.DOUBLE], dest=0, tag=1)


if __name__ == "__main__":
    main()
