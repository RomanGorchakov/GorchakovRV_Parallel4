
#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
from pathlib import Path
import csv
import os


def auxiliary_arrays_determination(size, numprocs):
    """Распределение элементов между всеми процессами, включая процесс 0."""
    base, remainder = divmod(int(size), numprocs)

    counts = np.array(
        [base + (1 if rank < remainder else 0)
         for rank in range(numprocs)],
        dtype=np.int32
    )

    displs = np.zeros(numprocs, dtype=np.int32)

    if numprocs > 1:
        displs[1:] = np.cumsum(counts[:-1])

    return counts, displs


def allreduce_scalar(value, comm):
    """Суммирование скалярной величины по всем процессам."""
    send = np.array([value], dtype=np.float64)
    recv = np.zeros(1, dtype=np.float64)

    comm.Allreduce(send, recv, op=MPI.SUM)

    return float(recv[0])


def conjugate_gradient_method(
    A_part,
    b_part,
    x_part,
    N,
    counts_N,
    displs_N,
    comm,
    tol=1e-10,
    max_iter=None
):
    """
    Решение системы нормальных уравнений:
        (A.T @ A) x = A.T @ b

    A_part и b_part содержат строки матрицы и вектора,
    распределённые между процессами.
    x_part содержит часть искомого вектора x.
    """

    rank = comm.Get_rank()

    if max_iter is None:
        max_iter = max(100, 10 * N)

    x_part = x_part.copy()

    # Локальная часть правой стороны: A_part.T @ b_part.
    rhs_local = A_part.T @ b_part

    # Суммирование вкладов всех процессов.
    rhs = np.zeros(N, dtype=np.float64)
    comm.Allreduce(rhs_local, rhs, op=MPI.SUM)

    # Каждый процесс хранит только свой участок x и остатка r.
    start = int(displs_N[rank])
    local_size = len(x_part)
    end = start + local_size

    x_part.fill(0.0)
    r_part = rhs[start:end].copy()
    p_part = r_part.copy()

    rr = allreduce_scalar(np.dot(r_part, r_part), comm)
    rhs_norm = np.sqrt(max(rr, 0.0))
    threshold = tol * max(1.0, rhs_norm)

    iterations = 0

    for iteration in range(max_iter):
        if np.sqrt(max(rr, 0.0)) <= threshold:
            break

        # Собираем полный вектор направления p на каждом процессе.
        p_full = np.empty(N, dtype=np.float64)

        comm.Allgatherv(
            [p_part, local_size, MPI.DOUBLE],
            [p_full, counts_N, displs_N, MPI.DOUBLE]
        )

        # Локальный вклад в (A.T @ A) @ p.
        Ap_part = A_part @ p_full
        q_local = A_part.T @ Ap_part

        # Полный вектор q = A.T @ A @ p.
        q_full = np.zeros(N, dtype=np.float64)
        comm.Allreduce(q_local, q_full, op=MPI.SUM)

        q_part = q_full[start:end]

        # p.T @ q: суммируем локальные скалярные произведения.
        p_dot_q = allreduce_scalar(
            np.dot(p_part, q_part),
            comm
        )

        # Для корректного шага требуется положительное p.T @ q.
        if not np.isfinite(p_dot_q) or p_dot_q <= 0.0:
            break

        alpha = rr / p_dot_q

        x_part += alpha * p_part
        r_part -= alpha * q_part

        rr_new = allreduce_scalar(
            np.dot(r_part, r_part),
            comm
        )

        iterations = iteration + 1

        if np.sqrt(max(rr_new, 0.0)) <= threshold:
            rr = rr_new
            break

        if rr <= 0.0:
            rr = rr_new
            break

        beta = rr_new / rr
        p_part = r_part + beta * p_part
        rr = rr_new

    residual_norm = np.sqrt(max(rr, 0.0))

    return x_part, iterations, residual_norm


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    data_dir = Path("data")

    # Читаем размеры задачи на процессе 0.
    if rank == 0:
        with (data_dir / "in.dat").open("r") as f:
            N = int(f.readline().strip())
            M = int(f.readline().strip())

        dimensions = np.array([N, M], dtype=np.int32)
    else:
        dimensions = np.empty(2, dtype=np.int32)

    comm.Bcast(dimensions, root=0)
    N, M = map(int, dimensions)

    if N <= 0 or M <= 0:
        raise ValueError("N и M должны быть положительными числами.")

    # Распределяем строки A и элементы x между всеми процессами.
    counts_M, displs_M = auxiliary_arrays_determination(
        M, numprocs
    )
    counts_N, displs_N = auxiliary_arrays_determination(
        N, numprocs
    )

    counts_A = (counts_M * N).astype(np.int32)
    displs_A = (displs_M * N).astype(np.int32)

    local_rows = int(counts_M[rank])
    local_x_size = int(counts_N[rank])

    A_part = np.empty((local_rows, N), dtype=np.float64)
    b_part = np.empty(local_rows, dtype=np.float64)
    x_part = np.zeros(local_x_size, dtype=np.float64)

    # Синхронизируем процессы перед началом измерения.
    comm.Barrier()
    start_time = MPI.Wtime()

    # Процесс 0 загружает исходные данные.
    if rank == 0:
        A = np.loadtxt(
            data_dir / "AData.dat",
            dtype=np.float64
        ).reshape(M, N)

        b = np.loadtxt(
            data_dir / "bData.dat",
            dtype=np.float64
        ).reshape(M)

        A_flat = np.ascontiguousarray(A.ravel())

        sendbuf_A = [
            A_flat, counts_A, displs_A, MPI.DOUBLE
        ]
        sendbuf_b = [
            b, counts_M, displs_M, MPI.DOUBLE
        ]
    else:
        A = None
        b = None
        sendbuf_A = None
        sendbuf_b = None

    # Распределяем строки матрицы A и вектора b.
    comm.Scatterv(
        sendbuf_A,
        [A_part.ravel(), local_rows * N, MPI.DOUBLE],
        root=0
    )

    comm.Scatterv(
        sendbuf_b,
        [b_part, local_rows, MPI.DOUBLE],
        root=0
    )

    # Решаем систему методом сопряжённых градиентов.
    x_part, iterations, residual_norm = conjugate_gradient_method(
        A_part,
        b_part,
        x_part,
        N,
        counts_N,
        displs_N,
        comm
    )

    # Собираем части решения на процессе 0.
    if rank == 0:
        x_parallel = np.empty(N, dtype=np.float64)
        recvbuf_x = [
            x_parallel, counts_N, displs_N, MPI.DOUBLE
        ]
    else:
        x_parallel = None
        recvbuf_x = None

    comm.Gatherv(
        [x_part, local_x_size, MPI.DOUBLE],
        recvbuf_x,
        root=0
    )

    # Все процессы завершают измеряемую часть синхронно.
    comm.Barrier()
    elapsed_time = MPI.Wtime() - start_time

    if rank == 0:
        # Проверка решения выполняется вне измеряемого времени.
        x_reference = np.linalg.lstsq(A, b, rcond=None)[0]

        difference = np.linalg.norm(x_parallel - x_reference)
        relative_difference = difference / max(
            1.0, np.linalg.norm(x_reference)
        )

        residual_parallel = np.linalg.norm(A @ x_parallel - b)
        residual_reference = np.linalg.norm(A @ x_reference - b)

        # Сохраняем найденный вектор.
        np.savetxt(
            "results/x_parallel_2.dat",
            x_parallel,
            fmt="%.16e"
        )

        print(f"N = {N}, M = {M}")
        print(f"Number of iterations: {iterations}")
        print(f"Residual norm of normal equations: {residual_norm:.6e}")
        print(
            f"||A*x_parallel - b||_2: "
            f"{residual_parallel:.6e}"
        )
        print(f"Residual norm of NumPy: {residual_reference:.6e}")
        print(
            f"Relative difference between solutions: "
            f"{relative_difference:.6e}"
        )
        print(f"Execution time: {elapsed_time:.6f} seconds")

        # Записываем результаты измерения в CSV.
        timing_file = "results/timings_3_2.csv"
        file_exists = (
            os.path.exists(timing_file)
            and os.path.getsize(timing_file) > 0
        )

        with open(timing_file, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "processes",
                    "N",
                    "M",
                    "time_seconds",
                    "iterations",
                    "normal_equation_residual",
                    "least_squares_residual",
                    "relative_difference"
                ])

            writer.writerow([
                numprocs,
                N,
                M,
                f"{elapsed_time:.6f}",
                iterations,
                f"{residual_norm:.6e}",
                f"{residual_parallel:.6e}",
                f"{relative_difference:.6e}"
            ])


if __name__ == "__main__":
    main()