
#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np


def read_params(filename="data/in.dat"):
    """Чтение размеров матрицы M x N."""
    with open(filename, "r") as f:
        M = int(f.readline().strip())
        N = int(f.readline().strip())

    return M, N


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
    size = comm.Get_size()

    # Чтение размеров матрицы на процессе 0
    if rank == 0:
        M_val, N_val = read_params("data/in.dat")
        params = np.array([M_val, N_val], dtype=np.int32)
    else:
        params = np.empty(2, dtype=np.int32)

    # Рассылка параметров всем процессам
    comm.Bcast([params, MPI.INT], root=0)
    M, N = int(params[0]), int(params[1])

    # Распределение строк матрицы
    rcounts, displs = auxiliary_arrays_determination(M, size)

    local_M = int(rcounts[rank])
    local_elements = local_M * N

    # Выделение локальных буферов
    A_loc = np.empty((local_M, N), dtype=np.float64)
    x_loc = np.empty(local_M, dtype=np.float64)
    b_loc = np.empty(N, dtype=np.float64)

    # Синхронизация перед измерением времени
    comm.Barrier()
    start_time = MPI.Wtime()

    # Чтение матрицы A на процессе 0
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

    # Распределение строк матрицы через Scatterv
    comm.Scatterv(
        sendbuf_A,
        [A_loc.ravel(), local_elements, MPI.DOUBLE],
        root=0
    )

    # Чтение вектора x длины M на процессе 0
    if rank == 0:
        x = np.empty(M, dtype=np.float64)

        with open("data/xData.dat", "r") as f:
            for i in range(M):
                x[i] = float(f.readline())

        sendbuf_x = [x, rcounts, displs, MPI.DOUBLE]
    else:
        x = None
        sendbuf_x = None

    # Распределение элементов вектора через Scatterv
    comm.Scatterv(
        sendbuf_x,
        [x_loc, local_M, MPI.DOUBLE],
        root=0
    )

    # Локальное вычисление b_loc = A_loc.T * x_loc
    np.dot(A_loc.T, x_loc, out=b_loc)

    # Сбор локальных результатов суммированием через Reduce
    if rank == 0:
        b = np.empty(N, dtype=np.float64)
        recvbuf_b = [b, MPI.DOUBLE]
    else:
        b = None
        recvbuf_b = None

    comm.Reduce(
        [b_loc, MPI.DOUBLE],
        recvbuf_b,
        op=MPI.SUM,
        root=0
    )

    # Завершение замера после сбора результатов
    comm.Barrier()
    elapsed_time = MPI.Wtime() - start_time

    # Проверка результата и сохранение статистики
    if rank == 0:
        # Последовательное вычисление для проверки
        b_seq = np.dot(A.T, x)

        diff_norm = np.linalg.norm(b - b_seq)
        correct = np.allclose(
            b, b_seq, rtol=1e-10, atol=1e-8
        )

        print(f"Differential norm: {diff_norm:.6e}")
        print(f"Execution time: {elapsed_time:.6f} seconds")

        # Запись итогового вектора
        with open("results/Results_parallel_atx.dat", "w") as f:
            for value in b:
                f.write(f"{value}\n")

        # Запись параметров эксперимента
        with open("results/timings_2_2.csv", "a") as f:
            f.write(
                f"{size},{M},{N},{elapsed_time:.6f}\n"
            )


if __name__ == "__main__":
    main()