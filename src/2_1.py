#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys


def make_counts_displs(M, size):
    """Расчет количества элементов и смещений для каждого процесса."""
    base, extra = divmod(M, size)

    rcounts = np.array(
        [base + (1 if rank < extra else 0) for rank in range(size)],
        dtype=np.int32
    )

    displs = np.zeros(size, dtype=np.int32)
    if size > 1:
        displs[1:] = np.cumsum(rcounts[:-1])

    return rcounts, displs


def main():
    # Инициализация MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Чтение размера вектора на процессе 0
    if rank == 0:
        try:
            if len(sys.argv) != 2:
                raise ValueError

            M = int(sys.argv[1])
            if M <= 0:
                raise ValueError

        except ValueError:
            print(
                "Использование: mpiexec -n <proc> "
                "python3 2_1.py <M>, где M > 0"
            )
            M = 0
    else:
        M = None

    # Рассылка размера вектора всем процессам
    M = comm.bcast(M, root=0)

    if M <= 0:
        return

    # Определение распределения элементов
    rcounts, displs = make_counts_displs(M, size)
    local_n = int(rcounts[rank])

    # Выделение локального буфера
    a_local = np.empty(local_n, dtype=np.float64)

    # Синхронизация перед измерением времени
    comm.Barrier()
    start_time = MPI.Wtime()

    # Генерация вектора на процессе 0
    if rank == 0:
        a = np.arange(1, M + 1, dtype=np.float64)
        sendbuf = [a, rcounts, displs, MPI.DOUBLE]
    else:
        a = None
        sendbuf = None

    # Распределение вектора между процессами
    comm.Scatterv(
        sendbuf,
        [a_local, MPI.DOUBLE],
        root=0
    )

    # Локальное вычисление скалярного произведения
    local_dot = np.array(
        [np.dot(a_local, a_local)],
        dtype=np.float64
    )

    # Измерение времени операции Reduce
    comm.Barrier()
    reduce_start = MPI.Wtime()

    if rank == 0:
        global_dot_reduce = np.empty(1, dtype=np.float64)
    else:
        global_dot_reduce = None

    comm.Reduce(
        [local_dot, MPI.DOUBLE],
        [global_dot_reduce, MPI.DOUBLE] if rank == 0 else None,
        op=MPI.SUM,
        root=0
    )

    reduce_time = MPI.Wtime() - reduce_start

    # Измерение времени операции Allreduce
    comm.Barrier()
    allreduce_start = MPI.Wtime()

    global_dot_allreduce = np.empty(1, dtype=np.float64)

    comm.Allreduce(
        [local_dot, MPI.DOUBLE],
        [global_dot_allreduce, MPI.DOUBLE],
        op=MPI.SUM
    )

    allreduce_time = MPI.Wtime() - allreduce_start

    # Завершение общего измерения
    comm.Barrier()
    elapsed_time = MPI.Wtime() - start_time

    # Проверка результата и сохранение статистики
    if rank == 0:
        sequential = np.dot(a, a)

        result_reduce = global_dot_reduce[0]
        result_allreduce = global_dot_allreduce[0]

        ok_reduce = np.isclose(
            sequential, result_reduce, rtol=1e-12, atol=1e-8
        )
        ok_allreduce = np.isclose(
            sequential, result_allreduce, rtol=1e-12, atol=1e-8
        )

        print(f"Sequential result = {sequential:.12g}")
        print(f"Reduce = {result_reduce:.12g}")
        print(f"Allreduce = {result_allreduce:.12g}")
        print(f"Reduce correct? {'YES' if ok_reduce else 'NO'}")
        print(f"Allreduce correct? {'YES' if ok_allreduce else 'NO'}")
        print(f"Reduce time: {reduce_time:.6f} seconds")
        print(f"Allreduce time: {allreduce_time:.6f} seconds")
        print(f"Total time: {elapsed_time:.6f} seconds")

        # Сохранение результатов эксперимента
        with open("results/timings_2_1.csv", "a") as f:
            f.write(
                f"{size},{M},{elapsed_time:.6f},"
                f"{reduce_time:.6f},{allreduce_time:.6f}\n"
            )


if __name__ == "__main__":
    main()