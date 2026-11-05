
#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from mpi4py import MPI


def auxiliary_arrays_determination(size, numprocs):
    """Распределение элементов между всеми MPI-процессами."""
    base, rem = divmod(int(size), numprocs)

    counts = np.array(
        [base + (1 if i < rem else 0) for i in range(numprocs)],
        dtype=np.intc
    )

    displs = np.zeros(numprocs, dtype=np.intc)
    if numprocs > 1:
        displs[1:] = np.cumsum(counts[:-1])

    return counts, displs


def allreduce_scalar(value, comm):
    """Суммирование скалярной величины между всеми процессами."""
    send = np.array([value], dtype=np.float64)
    recv = np.zeros(1, dtype=np.float64)

    comm.Allreduce(send, recv, op=MPI.SUM)
    return float(recv[0])


def conjugate_gradient_method(
    A_part, b_part, x_part, N,
    rcounts_N, displs_N, comm,
    tol=1e-10, max_iter=None
):
    """
    Метод сопряженных градиентов для системы нормальных уравнений:
        (A.T @ A) x = A.T @ b.

    Матрица A и вектор b распределены по строкам.
    Векторы x, r и p распределены по элементам.
    """
    rank = comm.Get_rank()

    if max_iter is None:
        max_iter = max(100, 10 * N)

    n_local = len(x_part)
    x_part = x_part.copy()

    # Правая часть нормальных уравнений: A.T @ b
    rhs_local = A_part.T @ b_part
    rhs = np.zeros(N, dtype=np.float64)
    comm.Allreduce(rhs_local, rhs, op=MPI.SUM)

    # Начальное приближение x = 0
    x_part.fill(0.0)

    start = int(displs_N[rank])
    end = start + n_local

    r_part = rhs[start:end].copy()
    p_part = r_part.copy()

    rr = allreduce_scalar(np.dot(r_part, r_part), comm)
    rhs_norm = np.sqrt(rr)
    threshold = tol * max(1.0, rhs_norm)

    iterations = 0

    for iteration in range(max_iter):
        if np.sqrt(max(rr, 0.0)) <= threshold:
            break

        # Сбор полного направления p на каждом процессе
        p_full = np.empty(N, dtype=np.float64)
        comm.Allgatherv(
            [p_part, n_local, MPI.DOUBLE],
            [p_full, rcounts_N, displs_N, MPI.DOUBLE]
        )

        # Локальный вклад в A.T @ A @ p
        Ap_part = A_part @ p_full
        q_local = A_part.T @ Ap_part

        q_full = np.zeros(N, dtype=np.float64)
        comm.Allreduce(q_local, q_full, op=MPI.SUM)

        q_part = q_full[start:end]

        p_dot_q = allreduce_scalar(
            np.dot(p_part, q_part), comm
        )

        if not np.isfinite(p_dot_q) or p_dot_q <= 0.0:
            break

        alpha = rr / p_dot_q

        x_part += alpha * p_part
        r_part -= alpha * q_part

        rr_new = allreduce_scalar(
            np.dot(r_part, r_part), comm
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

    # Чтение параметров задачи
    if rank == 0:
        data_dir = Path("data")

        with (data_dir / "in.dat").open("r") as f:
            N = int(f.readline().strip())
            M = int(f.readline().strip())

        dims = np.array([N, M], dtype=np.intc)
    else:
        dims = np.empty(2, dtype=np.intc)

    comm.Bcast(dims, root=0)
    N, M = map(int, dims)

    if N <= 0 or M <= 0:
        raise ValueError(
            f"Размеры должны быть положительными: N={N}, M={M}"
        )

    # Распределение строк матрицы и элементов вектора решения
    rcounts_M, displs_M = auxiliary_arrays_determination(
        M, numprocs
    )
    rcounts_N, displs_N = auxiliary_arrays_determination(
        N, numprocs
    )

    sendcounts_A = (rcounts_M * N).astype(np.intc)
    displs_A = (displs_M * N).astype(np.intc)

    n_rows_local = int(rcounts_M[rank])
    n_x_local = int(rcounts_N[rank])

    A_part = np.empty(
        (n_rows_local, N), dtype=np.float64
    )
    b_part = np.empty(n_rows_local, dtype=np.float64)
    x_part = np.zeros(n_x_local, dtype=np.float64)

    # Синхронизация перед измерением времени
    comm.Barrier()
    start_time = MPI.Wtime()

    # Чтение матрицы A и вектора b на процессе 0
    if rank == 0:
        data_dir = Path("data")

        A = np.loadtxt(
            data_dir / "AData.dat", dtype=np.float64
        ).reshape(M, N)

        b = np.loadtxt(
            data_dir / "bData.dat", dtype=np.float64
        ).reshape(M)

        A_flat = np.ascontiguousarray(A.ravel())

        sendbuf_A = [
            A_flat, sendcounts_A, displs_A, MPI.DOUBLE
        ]
        sendbuf_b = [
            b, rcounts_M, displs_M, MPI.DOUBLE
        ]
    else:
        A = None
        b = None
        sendbuf_A = None
        sendbuf_b = None

    # Распределение матрицы и правой части
    comm.Scatterv(
        sendbuf_A,
        [A_part.ravel(), n_rows_local * N, MPI.DOUBLE],
        root=0
    )

    comm.Scatterv(
        sendbuf_b,
        [b_part, n_rows_local, MPI.DOUBLE],
        root=0
    )

    # Параллельный метод сопряженных градиентов
    x_part, iterations, residual_norm = conjugate_gradient_method(
        A_part,
        b_part,
        x_part,
        N,
        rcounts_N,
        displs_N,
        comm
    )

    # Сбор окончательного решения
    if rank == 0:
        x_parallel = np.empty(N, dtype=np.float64)
        recvbuf_x = [
            x_parallel, rcounts_N, displs_N, MPI.DOUBLE
        ]
    else:
        x_parallel = None
        recvbuf_x = None

    comm.Gatherv(
        [x_part, n_x_local, MPI.DOUBLE],
        recvbuf_x,
        root=0
    )

    # Завершение измерения после сбора решения
    comm.Barrier()
    elapsed_time = MPI.Wtime() - start_time

    # Проверка и сохранение результатов на процессе 0
    if rank == 0:
        x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]

        difference = np.linalg.norm(x_parallel - x_numpy)
        relative_difference = difference / max(
            1.0, np.linalg.norm(x_numpy)
        )

        residual_parallel = np.linalg.norm(
            A @ x_parallel - b
        )
        residual_numpy = np.linalg.norm(
            A @ x_numpy - b
        )

        np.savetxt(
            "results/x_parallel_1.dat",
            x_parallel,
            fmt="%.16e"
        )

        print(f"Number of iterations: {iterations}")
        print(f"Residual norm of normal equations: {residual_norm:.6e}")
        print(
            f"||A*x_parallel - b||_2: "
            f"{residual_parallel:.6e}"
        )
        print(
            f"||A*x_numpy - b||_2: "
            f"{residual_numpy:.6e}"
        )
        print(
            f"||x_parallel - x_numpy||_2: "
            f"{difference:.6e}"
        )
        print(
            f"Relative difference between solutions: "
            f"{relative_difference:.6e}"
        )
        print(
            f"Execution time: {elapsed_time:.6f} seconds"
        )

        # Saving experiment results
        with open("results/timings_3_1.csv", "a") as f:
            f.write(
                f"{numprocs},{N},{M},{elapsed_time:.6f},"
                f"{iterations},{residual_norm:.6e}\n"
            )


if __name__ == "__main__":
    main()