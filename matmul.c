#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>

#define RAND_INT(max) (rand() % (max))
#define IDX(i,j,n)   ((i)*(n) + (j))

typedef struct {
    int tid;
    int n;
    int row_start;
    int row_end;
    const int *A;
    const int *B;
    int       *C;
} thread_arg_t;

static void matmul_range(const int *A, const int *B, int *C, int n, int r0, int r1){
    for (int i = r0; i < r1; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[IDX(i,k,n)] * B[IDX(k,j,n)];
            C[IDX(i,j,n)] = sum;
        }
    }
}

static void *worker(void *arg_ptr)
{
    thread_arg_t *arg = (thread_arg_t *)arg_ptr;
    matmul_range(arg->A, arg->B, arg->C, arg->n, arg->row_start, arg->row_end);
    return NULL;
}

static inline long long diff_ns(struct timespec a, struct timespec b)
{
    return (long long)(b.tv_sec - a.tv_sec) * 1000000000LL + (long long)(b.tv_nsec - a.tv_nsec);
}

static int *alloc_mat(int n)
{
    int *m = (int *)malloc((size_t)n * n * sizeof(int));
    if (!m) { perror("malloc"); exit(1);} ;
    return m;
}

static void rand_fill(int *m, int n)
{
    for (int i = 0; i < n*n; i++){
        m[i] = RAND_INT(10);
    }
}

static int compare(const int *X, const int *Y, int n)
{
    for (int i = 0; i < n*n; i++){
        if (X[i] != Y[i]){
            return -1;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <n> [trials]\n", argv[0]);
        return 1;
    }
    int n      = atoi(argv[1]);
    int trials = (argc > 2) ? atoi(argv[2]) : 1;
    if (n <= 0 || trials <= 0) {
        fprintf(stderr, "n and trials must be positive\n");
        return 1;
    }

    srand((unsigned)time(NULL));

    char fname[32];
    snprintf(fname, sizeof(fname), "%d.out", n);
    FILE *fp = fopen(fname, "a");
    if (!fp) { perror("fopen"); exit(1);} ;

    int *A = alloc_mat(n);
    int *B = alloc_mat(n);
    int *C_seq = alloc_mat(n);
    int *C_par = alloc_mat(n);

    for (int t = 0; t < trials; t++) {
        rand_fill(A, n);
        rand_fill(B, n);

        struct timespec ts0, ts1;

        /* sequential */
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
        matmul_range(A, B, C_seq, n, 0, n);
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        long long dt_seq = diff_ns(ts0, ts1);

        /* two threads */
        int mid = n / 2;
        pthread_t tid[2];
        thread_arg_t arg[2] = {
            {0, n, 0,   mid, A, B, C_par},
            {1, n, mid, n,   A, B, C_par}
        };
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
        for (int i = 0; i < 2; i++)
            if (pthread_create(&tid[i], NULL, worker, &arg[i]) != 0) {
                perror("pthread_create"); exit(1);
            }
        for (int i = 0; i < 2; i++)
            pthread_join(tid[i], NULL);
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        long long dt_two = diff_ns(ts0, ts1);

        /* verify */
        if (compare(C_seq, C_par, n) != 0) {
            fprintf(stderr, "mismatch in 2‑thread result\n");
            exit(1);
        }

        /* n threads (1 thread/row) ----- */
        pthread_t *tids = (pthread_t *)malloc((size_t)n * sizeof(pthread_t));
        thread_arg_t *args = (thread_arg_t *)malloc((size_t)n * sizeof(thread_arg_t));
        if (!tids || !args) { perror("malloc"); exit(1);} ;

        clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
        for (int i = 0; i < n; i++) {
            args[i].tid = i;
            args[i].n   = n;
            args[i].row_start = i;
            args[i].row_end   = i + 1;
            args[i].A = A;
            args[i].B = B;
            args[i].C = C_par; /* reuse buffer */
            if (pthread_create(&tids[i], NULL, worker, &args[i]) != 0) {
                perror("pthread_create"); exit(1);
            }
        }
        for (int i = 0; i < n; i++){
            pthread_join(tids[i], NULL);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        long long dt_n = diff_ns(ts0, ts1);

        if (compare(C_seq, C_par, n) != 0) {
            fprintf(stderr, "mismatch in n‑thread result\n");
            exit(1);
        }

        /* write timings */
        fprintf(fp, "%lld %lld %lld\n", dt_seq, dt_two, dt_n);

        free(tids);
        free(args);
    }

    fclose(fp);
    free(A); free(B); free(C_seq); free(C_par);
    return 0;
}

