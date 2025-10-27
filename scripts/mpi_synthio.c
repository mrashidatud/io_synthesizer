/* mpi_synthio.c
 *
 * Harness: deterministic Consec → SeqRemainder → Random per phase,
 *          with L→M→S order driven by the planner's CSV.
 *
 * Enforced behavior:
 *   - MPI job so Darshan records MPI context.
 *   - DATA phases (pread/pwrite only; no lseek for data):
 *       * p_ua_file: for xfer >= FILE_ALIGN, enforce the exact fraction of NOT-aligned starts
 *         (unaligned quota via +1B skew; the rest snapped to align).
 *       * p_ua_mem: per-op buffer misalignment by 1B for the planned fraction.
 *       * Consec: adjacent; Seq remainder: fixed gap=xfer; Random: descending 128MiB chunks
 *         with RR across chunks; optional pre_seek_eof fence.
 *       * Offsets clamped into [0, size-xfer]; files pre-truncated by planner.
 *   - META-only phase drives open/stat/seek/sync counts in a single phase.
 *
 * Multi-rank shared/unique:
 *   - "|shared|" → all ranks participate (ops sharded equally) → Darshan SHARED.
 *   - "|unique|" → exactly one owner rank (stable hash(path)) executes; others DO NOT OPEN → UNIQUE.
 *   - No marker → legacy: rank 0 executes (others no-op).
 *
 * RW switches:
 *   - Planner alternates rows on the same path; no extra logic here.
 *
 * Build:
 *   mpicc -O3 -Wall -Wextra -o mpi_synthio mpi_synthio.c -lm
 *
 * Run:
 *   mpiexec -n <N> ./mpi_synthio --plan payload/plan.csv \
 *           --io-api posix --meta-api posix --collective none
 */

#define _GNU_SOURCE
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifndef LUSTRE_FILE_ALIGN
#define LUSTRE_FILE_ALIGN (1<<20) /* 1 MiB default; planner sets alignment policy via p_ua_file */
#endif

#ifndef MEM_ALIGN
#define MEM_ALIGN 8
#endif

#define MAX_LINE  4096
#define CHUNK_RANDOM (128*1024*1024)
#define TSBUF_LEN 64

typedef enum { API_POSIX=0, API_MPIIO=1 } io_api_t;
typedef enum { META_POSIX=0 } meta_api_t;
typedef enum { COL_NONE=0, COL_INDEP=1, COL_COLL=2 } col_mode_t;

typedef struct {
    char   type[8];    /* "data" or "meta" */
    char   path[1024];
    uint64_t total_bytes;
    uint64_t xfer;
    double p_write;
    double p_rand;
    double p_seq_r;
    double p_consec_r;
    double p_seq_w;
    double p_consec_w;
    double p_ua_file;   /* fraction of ops with NOT aligned file start (planner-proportional) */
    double p_ua_mem;    /* fraction of ops with NOT aligned memory buffer (planner-proportional) */
    double rw_switch;   /* unused here; planner creates switches via row alternation */
    uint64_t meta_open, meta_stat, meta_seek, meta_sync;
    uint32_t seed;
    char   flags[256];  /* "consec|shared|" etc. Optional "|shared|" or "|unique|" marker. */
    double p_rand_fwd_r, p_rand_fwd_w, p_consec_internal; /* reserved */
    int    pre_seek_eof; /* 0/1 */
    uint64_t n_ops;      /* optional: planned ops in this phase (info only) */
} phase_t;

static int startswith(const char*s,const char*p){ return strncmp(s,p,strlen(p))==0; }

/* Parse a CSV line that may contain the optional pre_seek_eof and n_ops columns. */
static int parse_line(const char* line, phase_t* ph)
{
    char buf[MAX_LINE];
    strncpy(buf,line,MAX_LINE-1); buf[MAX_LINE-1]=0;

    if(startswith(buf,"type,")) return 1; /* header */

    char *tok; int col=0;
    memset(ph,0,sizeof(*ph));
    ph->pre_seek_eof = 0;
    ph->n_ops = 0;

    tok = strtok(buf, ",\n");
    while(tok){
        switch(col){
            case 0:  strncpy(ph->type, tok, sizeof(ph->type)-1); break;
            case 1:  strncpy(ph->path, tok, sizeof(ph->path)-1); break;
            case 2:  ph->total_bytes = strtoull(tok,NULL,10); break;
            case 3:  ph->xfer = strtoull(tok,NULL,10); break;
            case 4:  ph->p_write = atof(tok); break;
            case 5:  ph->p_rand = atof(tok); break;
            case 6:  ph->p_seq_r = atof(tok); break;
            case 7:  ph->p_consec_r = atof(tok); break;
            case 8:  ph->p_seq_w = atof(tok); break;
            case 9:  ph->p_consec_w = atof(tok); break;
            case 10: ph->p_ua_file = atof(tok); break;
            case 11: ph->p_ua_mem  = atof(tok); break;
            case 12: ph->rw_switch = atof(tok); break;
            case 13: ph->meta_open = strtoull(tok,NULL,10); break;
            case 14: ph->meta_stat = strtoull(tok,NULL,10); break;
            case 15: ph->meta_seek = strtoull(tok,NULL,10); break;
            case 16: ph->meta_sync = strtoull(tok,NULL,10); break;
            case 17: ph->seed = (uint32_t)strtoul(tok,NULL,10); break;
            case 18: strncpy(ph->flags, tok, sizeof(ph->flags)-1); break;
            case 19: ph->p_rand_fwd_r = atof(tok); break;
            case 20: ph->p_rand_fwd_w = atof(tok); break;
            case 21: ph->p_consec_internal = atof(tok); break;
            case 22: ph->pre_seek_eof = atoi(tok); break; /* optional */
            case 23: ph->n_ops = strtoull(tok,NULL,10); break; /* optional */
            default: break;
        }
        tok = strtok(NULL, ",\n"); col++;
    }
    return 0;
}

typedef struct { void* base; void* ptr; } buf_t;

static buf_t alloc_buffer(size_t sz)
{
    buf_t b; b.base=NULL; b.ptr=NULL;
    if(posix_memalign(&b.base, MEM_ALIGN, sz + MEM_ALIGN)) return b;
    b.ptr = b.base;
    return b;
}

static off_t clamp_range(off_t start, off_t file_size, uint64_t xfer)
{
    if(start < 0) start = 0;
    off_t max = (off_t)file_size - (off_t)xfer;
    if(max < 0) max = 0;
    if(start > max) start = max;
    return start;
}

static ssize_t do_prw(int fd2, void* buf2, size_t len, off_t off, int wr)
{
    size_t left = len;
    while(left>0){
        ssize_t rc = wr ? pwrite(fd2, buf2, left, off) : pread(fd2, buf2, left, off);
        if(rc<0){ return rc; }
        buf2 = (void*)((uintptr_t)buf2 + (size_t)rc);
        off  += (off_t)rc;
        left -= (size_t)rc;
    }
    return (ssize_t)len;
}

/* FNV-1a 64-bit for stable owner selection of unique rows */
static unsigned long fnv1a_64(const char* s)
{
    unsigned long h = 1469598103934665603ULL;
    while(*s){
        h ^= (unsigned long)(unsigned char)(*s++);
        h *= 1099511628211ULL;
    }
    return h;
}

// === ADD: helpers for flags, time stamps, sharding, run detection ===
static inline int has_flag(const char* flags, const char* needle) {
    return flags && strstr(flags, needle) != NULL;
}
static inline int is_shared_row_flags(const char* flags) { return has_flag(flags, "|shared|"); }
static inline int is_unique_row_flags(const char* flags) { return has_flag(flags, "|unique|"); }

/* Wallclock helpers: get now, format ISO8601 UTC */
static inline void wallclock_now(struct timespec* ts) {
#if defined(CLOCK_REALTIME)
    clock_gettime(CLOCK_REALTIME, ts);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    ts->tv_sec = tv.tv_sec;
    ts->tv_nsec = (long)tv.tv_usec * 1000L;
#endif
}

static void wallclock_iso8601_from(struct timespec ts, char *out, size_t n)
{
    if (n < 32) {                    // safety: need ~28 bytes
        if (n) out[0] = '\0';
        return;
    }

    // round ns to nearest ms, clamp, and carry if needed
    long ns = ts.tv_nsec;
    if (ns < 0) ns = 0;
    unsigned ms = (unsigned)((ns + 500000L) / 1000000L);  // 0..1000

    time_t tt = ts.tv_sec;
    if (ms >= 1000U) {
        ms -= 1000U;
        tt += 1;                     // carry to next second
    }

    struct tm tm_utc;
    gmtime_r(&tt, &tm_utc);

    // YYYY-MM-DDTHH:MM:SS.mmmZ
    (void)snprintf(out, n, "%04d-%02d-%02dT%02d:%02d:%02d.%03uZ",
                   tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
                   tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec, ms);
}

// even split (for shared rows)
static inline uint64_t split_even(uint64_t total, int nprocs, int rank, uint64_t* start_index_out) {
    uint64_t base = total / (uint64_t)nprocs, rem = total % (uint64_t)nprocs;
    uint64_t mine = base + ((uint64_t)rank < rem ? 1 : 0);
    uint64_t start =
        ((uint64_t)rank < rem)
        ? ((uint64_t)rank * (base + 1))
        : (rem * (base + 1) + ((uint64_t)rank - rem) * base);
    if (start_index_out) *start_index_out = start;
    return mine;
}

// count a consecutive run of UNIQUE rows starting at idx
static int count_unique_run(const phase_t* phases, int nphases, int idx) {
    int k = 0;
    while (idx + k < nphases && is_unique_row_flags(phases[idx + k].flags)) ++k;
    return k;
}

static int owner_for_path(const char* path, int nprocs){
    return (int)(fnv1a_64(path) % (unsigned long)nprocs);
}

/* --- helpers for alignment (C) --- */
static inline off_t force_aligned_start(off_t s)
{
    off_t a = (s + (LUSTRE_FILE_ALIGN-1)) & ~(LUSTRE_FILE_ALIGN-1);
    return a;
}
static inline off_t force_unaligned_start(off_t s)
{
    if ((s % LUSTRE_FILE_ALIGN) == 0) s += 1; /* skew by 1B */
    return s;
}

// === ADD: execute a DATA row for exactly Nops_local ops starting at logical op-index 'start_idx'
static void exec_phase_data_local(const phase_t* p, int phase_idx,
                                  uint64_t Nops_local, uint64_t start_idx) {
    if (p->xfer == 0 || Nops_local == 0) return;

    int world_rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Open (participants only)
    int is_write = (p->p_write > 0.5);
    int fd = open(p->path, is_write ? O_RDWR|O_CREAT : O_RDONLY, 0644);
    if (fd < 0) { perror("open"); return; }

    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return; }
    off_t fsz = st.st_size;

    struct timespec ts_start, ts_end;
    wallclock_now(&ts_start);

    // Decide consec/seq/random counts the same way as in do_posix_data, but using Nops_local.
    double p_con, p_seq;
    if (is_write) { p_con = p->p_consec_w; p_seq = p->p_seq_w; }
    else          { p_con = p->p_consec_r; p_seq = p->p_seq_r; }
    if (p_con < 0.0) { p_con = 0.0; }
    if (p_con > 1.0) { p_con = 1.0; }
    if (p_seq < 0.0) { p_seq = 0.0; }
    if (p_seq > 1.0) { p_seq = 1.0; }
    if (p_seq < p_con) { p_seq = p_con; }

    uint64_t N_con = (uint64_t) llround((double)Nops_local * p_con);
    uint64_t N_seq = (uint64_t) llround((double)Nops_local * (p_seq - p_con));
    if (N_con > Nops_local) N_con = Nops_local;
    if (N_seq > Nops_local - N_con) N_seq = Nops_local - N_con;
    uint64_t N_rnd = Nops_local - N_con - N_seq;
    if ((p_seq > p_con) && N_seq == 0 && N_rnd > 0) { N_seq++; N_rnd--; }
    if (p_con > 0.0 && N_con == 0 && N_rnd > 0)     { N_con++; N_rnd--; }

    // Large-op alignment quotas
    int large = (p->xfer >= LUSTRE_FILE_ALIGN);
    uint64_t N_unaligned_total = 0, N_aligned_total = 0;
    if (large) {
        N_unaligned_total = (uint64_t)llround(p->p_ua_file * (double)Nops_local);
        if (N_unaligned_total > Nops_local) N_unaligned_total = Nops_local;
        N_aligned_total = Nops_local - N_unaligned_total;
    }
    uint64_t unaligned_left = N_unaligned_total;
    uint64_t aligned_left   = N_aligned_total;

    // mem misalignment
    uint64_t N_mem_mis = (uint64_t)llround(p->p_ua_mem * (double)Nops_local);
    uint64_t mis_left = N_mem_mis;

    buf_t b = alloc_buffer((size_t)p->xfer);
    if (!b.base) { perror("alloc_buffer"); close(fd); return; }
    void* aligned_ptr = b.base;
    void* mis_ptr     = (void*)((uintptr_t)b.base + 1);

    // Seed randomness per rank for random-phase, stable across runs
    srand(p->seed + (unsigned int)world_rank * 1337u + (unsigned int)(start_idx % 7919u));

    // The execution body is identical to your existing do_posix_data(),
    // except we use local counters (N_con/N_seq/N_rnd) and keep its offset math.
    // --- paste the consecutive/seq/random loops from do_posix_data here verbatim ---
    //     (from: "/* ----- Consecutive ----- */" down to the final log+free+close)
    //     Replace the final fprintf with the expanded one below to add timestamps & alignment info.

    // ====== BEGIN: copy of your loops (unchanged apart from counters) ======
    off_t cur = 0;
    if (N_con > 0) {
        if (large) {
            uint64_t U = (unaligned_left > N_con) ? N_con : unaligned_left;
            uint64_t A = N_con - U;
            if (U > 0) {
                off_t start = force_unaligned_start(cur);
                start = clamp_range(start, fsz, p->xfer);
                for (uint64_t i = 0; i < U; i++) {
                    void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
                    if (mis_left>0) mis_left--;
                    if (do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write) < 0) { perror(is_write?"pwrite":"pread"); U = 0; break; }
                    start += (off_t)p->xfer;
                }
                cur = (off_t)start; unaligned_left -= U;
            }
            if (A > 0) {
                off_t start = force_aligned_start(cur);
                start = clamp_range(start, fsz, p->xfer);
                for (uint64_t i = 0; i < A; i++) {
                    void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
                    if (mis_left>0) mis_left--;
                    if (do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write) < 0) { perror(is_write?"pwrite":"pread"); break; }
                    start += (off_t)p->xfer;
                }
                cur = (off_t)start; if (aligned_left >= A) aligned_left -= A; else aligned_left = 0;
            }
        } else {
            off_t start = clamp_range(cur, fsz, p->xfer);
            for (uint64_t i=0; i<N_con; i++) {
                void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
                if (mis_left>0) mis_left--;
                if (do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write) < 0) { perror(is_write?"pwrite":"pread"); break; }
                start += (off_t)p->xfer;
            }
            cur = start;
        }
    }

    if (N_seq > 0) {
        if (large) {
            uint64_t U = (unaligned_left > N_seq) ? N_seq : unaligned_left;
            uint64_t A = N_seq - U;
            if (U > 0) {
                off_t start = force_unaligned_start(cur + (off_t)p->xfer);
                start = clamp_range(start, fsz, p->xfer);
                for (uint64_t i=0; i<U; i++) {
                    void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
                    if (mis_left>0) mis_left--;
                    if (do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write) < 0) { perror(is_write?"pwrite":"pread"); U = 0; break; }
                    start += (off_t)(p->xfer * 2);
                }
                cur = start - (off_t)p->xfer; unaligned_left -= U;
            }
            if (A > 0) {
                off_t start = force_aligned_start(cur + (off_t)p->xfer);
                start = clamp_range(start, fsz, p->xfer);
                for (uint64_t i=0; i<A; i++) {
                    void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
                    if (mis_left>0) mis_left--;
                    if (do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write) < 0) { perror(is_write?"pwrite":"pread"); break; }
                    start += (off_t)(p->xfer * 2);
                }
                cur = start - (off_t)p->xfer; if (aligned_left >= A) aligned_left -= A; else aligned_left = 0;
            }
        } else {
            off_t start = clamp_range(cur + (off_t)p->xfer, fsz, p->xfer);
            for (uint64_t i=0; i<N_seq; i++) {
                void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
                if (mis_left>0) mis_left--;
                if (do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write) < 0) { perror(is_write?"pwrite":"pread"); break; }
                start += (off_t)(p->xfer * 2);
            }
            cur = start - (off_t)p->xfer;
        }
    }

    if (N_rnd > 0 && p->pre_seek_eof) { (void)lseek(fd, fsz, SEEK_SET); }
    uint64_t nchunks = (fsz>0) ? ((fsz + CHUNK_RANDOM - 1) / CHUNK_RANDOM) : 1;
    uint64_t cidx = (nchunks>0 ? nchunks-1 : 0);

    for (uint64_t i=0; i<N_rnd; i++) {
        off_t chunk_start = (off_t)cidx * (off_t)CHUNK_RANDOM;
        off_t chunk_end   = chunk_start + (off_t)CHUNK_RANDOM;
        if (chunk_end > fsz) chunk_end = fsz;
        off_t span = chunk_end - chunk_start;
        if (span < (off_t)p->xfer) {
            if (cidx>0) { cidx--; i--; continue; }
            chunk_start = 0; chunk_end = fsz; span = chunk_end - chunk_start;
            if (span < (off_t)p->xfer) break;
        }
        off_t start = chunk_start;
        if (span > (off_t)p->xfer) {
            uint64_t r = (uint64_t)rand();
            off_t delta = (off_t)(r % (uint64_t)(span - (off_t)p->xfer + 1));
            start = chunk_start + delta;
        }
        if (large) {
            if (unaligned_left>0) { start = force_unaligned_start(start); unaligned_left--; }
            else if (aligned_left>0) {
                off_t aligned = start & ~(off_t)(LUSTRE_FILE_ALIGN-1);
                if (aligned < chunk_start) aligned = chunk_start;
                if (aligned + (off_t)p->xfer <= chunk_end) start = aligned;
                else start = force_aligned_start(start);
                aligned_left--;
            }
        }
        start = clamp_range(start, fsz, p->xfer);
        void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
        if (mis_left>0) mis_left--;
        if (do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write) < 0) { perror(is_write?"pwrite":"pread"); break; }
        if (cidx>0) cidx--; else cidx = nchunks-1;
    }
    // ====== END: copy of loops ======

    wallclock_now(&ts_end);
    char ts0[TSBUF_LEN], ts1[TSBUF_LEN];
    wallclock_iso8601_from(ts_start, ts0, sizeof ts0);
    wallclock_iso8601_from(ts_end,   ts1, sizeof ts1);

    // Timestamped, richer phase log (stderr)
    fprintf(stderr,
        "rank %d: phase %d done %s: path=%s xfer=%" PRIu64 "B ops=%" PRIu64
        " (consec=%" PRIu64 ", seq=%" PRIu64 ", rand=%" PRIu64 ") %s flags=%s "
        "file_align=%dB mem_align=%dB start_idx=%" PRIu64 " start=%s end=%s\n",
        world_rank, phase_idx,
        is_shared_row_flags(p->flags) ? "[SHARED]" :
        (is_unique_row_flags(p->flags) ? "[UNIQUE]" : "[LEGACY]"),
        p->path, p->xfer, Nops_local, N_con, N_seq, N_rnd,
        is_write ? "[WRITE]" : "[READ]", p->flags,
        (int)LUSTRE_FILE_ALIGN, (int)MEM_ALIGN, start_idx, ts0, ts1);
    fflush(stderr);

    free(b.base);
    close(fd);
}

static void do_posix_meta(const phase_t* p, int phase_idx)
{
    int world_rank=0, world_n=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_n);

    const int is_shared = (strstr(p->flags, "|shared|") != NULL);
    const int is_unique = (strstr(p->flags, "|unique|") != NULL);

    uint64_t open_local=0, stat_local=0, seek_local=0, sync_local=0;

    if (is_shared) {
        // shard global totals evenly across ranks
        #define SHARD(total) ({ \
            uint64_t _T=(total), _q=_T/(uint64_t)world_n, _r=_T%(uint64_t)world_n; \
            _q + ((uint64_t)world_rank < _r ? 1 : 0); })
        open_local = SHARD(p->meta_open);
        stat_local = SHARD(p->meta_stat);
        seek_local = SHARD(p->meta_seek);
        sync_local = SHARD(p->meta_sync);
        #undef SHARD
    } else if (is_unique) {
        // pick a deterministic owner rank for this path
        int owner = 0;
        unsigned long h = 1469598103934665603ULL;
        const char* s = p->path;
        while (*s) { h ^= (unsigned long)(unsigned char)(*s++); h *= 1099511628211ULL; }
        owner = (int)(h % (unsigned long)world_n);
        if (world_rank != owner) return;
        open_local = p->meta_open;
        stat_local = p->meta_stat;
        seek_local = p->meta_seek;
        sync_local = p->meta_sync;
    } else {
        // fallback: leader-only
        if (world_rank != 0) return;
        open_local = p->meta_open;
        stat_local = p->meta_stat;
        seek_local = p->meta_seek;
        sync_local = p->meta_sync;
    }
    
    struct timespec ts_start, ts_end;
    wallclock_now(&ts_start);

    for (uint64_t i=0;i<open_local;i++){
        int fd = open(p->path, O_RDONLY|O_CREAT, 0644);
        if (fd>=0) close(fd);
    }
    struct stat st;
    for (uint64_t i=0;i<stat_local;i++){
        (void)stat(p->path, &st);
    }
    int fd = open(p->path, O_RDONLY|O_CREAT, 0644);
    if (fd>=0){
        for (uint64_t i=0;i<seek_local;i++){
            (void)lseek(fd, (off_t)(i%4096), SEEK_SET);
        }
        for (uint64_t i=0;i<sync_local;i++){
            (void)fsync(fd);
        }
        close(fd);
    }

    // === REPLACE: do_posix_meta() final fprintf with timestamped one ===
    wallclock_now(&ts_end);
    char ts0[TSBUF_LEN], ts1[TSBUF_LEN];
    wallclock_iso8601_from(ts_start, ts0, sizeof ts0);
    wallclock_iso8601_from(ts_end,   ts1, sizeof ts1);
    
    fprintf(stderr,
        "rank %d: phase %d done [META]: path=%s open=%" PRIu64 " stat=%" PRIu64
        " seek=%" PRIu64 " sync=%" PRIu64 " start=%s end=%s\n",
        world_rank, phase_idx, p->path, open_local, stat_local, seek_local, sync_local, ts0, ts1);
    fflush(stderr);
}

// === REPLACE: run_plan with wavefront scheduler ===
static void run_plan(FILE* fp)
{
    // 1) Parse all rows first
    phase_t* phases = NULL; size_t cap = 0, nph = 0;
    char line[MAX_LINE];
    while (fgets(line, sizeof(line), fp)) {
        phase_t p;
        if (parse_line(line, &p)) continue; // skip header/empty
        if (nph == cap) {
            cap = cap ? cap * 2 : 256;
            phases = (phase_t*)realloc(phases, cap * sizeof(phase_t));
            if (!phases) { perror("realloc phases"); MPI_Abort(MPI_COMM_WORLD, 3); }
        }
        phases[nph++] = p;
    }

    int rank=0, nprocs=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t idx = 0;
    while (idx < nph) {
        phase_t* P = &phases[idx];

        if (!strcmp(P->type, "meta")) {
            // META: keep your existing behavior (sharded when |shared|, owner/leader otherwise)
            MPI_Barrier(MPI_COMM_WORLD);
            double t_start = MPI_Wtime();

            do_posix_meta(P, (int)(idx + 1));

            double t_end = MPI_Wtime();
            double min_start=0.0, max_end=0.0;
            MPI_Reduce(&t_start, &min_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&t_end,   &max_end,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) {
                fprintf(stdout, "phase %zu SUMMARY: META wall=%.6fs (min_start=%.6f, max_end=%.6f)\n",
                        idx+1, max_end - min_start, min_start, max_end);
                fflush(stdout);
            }
            idx++;
            continue;
        }

        // DATA rows:
        const int is_shared = is_shared_row_flags(P->flags);
        const int is_unique = is_unique_row_flags(P->flags);

        // Compute global ops for this row
        uint64_t Nops = (P->xfer ? (P->total_bytes / P->xfer) : 0);

        if (is_shared || (!is_unique && !is_shared)) {
            // ===== SHARED (or legacy) =====
            uint64_t start_idx_local = 0;
            uint64_t my_ops = split_even(Nops, nprocs, rank, &start_idx_local);

            // Align start, then measure per-rank start time
            MPI_Barrier(MPI_COMM_WORLD);
            double t_start = MPI_Wtime();

            exec_phase_data_local(P, (int)(idx + 1), my_ops, start_idx_local);

            double t_end = MPI_Wtime();

            // Compute global bounds: min start, max end, and sum of ops
            double min_start = 0.0, max_end = 0.0;
            uint64_t my_ops_u64 = my_ops;
            unsigned long long my_ops_ull = (unsigned long long)my_ops_u64;
            unsigned long long sum_ops_ull = 0ULL;

            MPI_Reduce(&t_start, &min_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&t_end,   &max_end,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&my_ops_ull, &sum_ops_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD); // ensure all finished before printing

            if (rank == 0) {
                double wall = max_end - min_start;
                fprintf(stdout,
                    "phase %zu SUMMARY: SHARED/LEGACY ops=%llu wall=%.6fs (min_start=%.6f, max_end=%.6f)\n",
                    idx + 1, sum_ops_ull, wall, min_start, max_end);
                fflush(stdout);
            }
            idx++;
            continue;
        }

        // ===== UNIQUE RUN in waves =====
        int run_len = count_unique_run(phases, (int)nph, (int)idx);

        /* done[i]==1 means phases[idx+i] has executed */
        char *done = (char*)malloc((size_t)run_len);
        if (!done) { fprintf(stderr, "OOM done\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < run_len; i++) done[i] = 0;

        int remaining = run_len;

        while (remaining > 0) {
            int  assigned = 0;
            int *chosen   = (int*)malloc(sizeof(int) * nprocs);   /* global index in [0..run_len), or -1 */
            if (!chosen) { fprintf(stderr, "OOM chosen\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
            for (int r = 0; r < nprocs; r++) chosen[r] = -1;

            if (rank == 0) {
                int *owner_used = (int*)malloc(sizeof(int) * nprocs);
                if (!owner_used) { fprintf(stderr, "OOM owner_used\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
                for (int r = 0; r < nprocs; r++) owner_used[r] = 0;

                /* Walk ALL rows in this UNIQUE run, earliest-first. Give each owner at most one. */
                for (int i = 0; i < run_len; i++) {
                    if (done[i]) continue;
                    phase_t *U = &phases[idx + i];
                    int owner  = owner_for_path(U->path, nprocs);
                    if (!owner_used[owner] && chosen[owner] == -1) {
                        chosen[owner]  = i;     /* this rank runs global row i this round */
                        owner_used[owner] = 1;
                        assigned++;
                        if (assigned == nprocs) break; /* all ranks got work */
                    }
                }
                free(owner_used);
            }

            /* Share the plan for this round */
            MPI_Bcast(&assigned, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(chosen,   nprocs, MPI_INT, 0, MPI_COMM_WORLD);

            /* All ranks mark the chosen rows as done (so state stays identical) */
            for (int r = 0; r < nprocs; r++) {
                if (chosen[r] >= 0) {
                    if (!done[chosen[r]]) {
                        done[chosen[r]] = 1;
                    }
                }
            }
            remaining -= assigned;

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            /* Each rank executes at most one row this round */
            if (chosen[rank] >= 0) {
                int gi = chosen[rank];                /* global index into the run */
                phase_t *U = &phases[idx + gi];

                /* derive Nops = total_bytes/xfer (planner semantics) */
                unsigned long long Nops = 0ULL;
                if (U->xfer > 0) Nops = (unsigned long long)(U->total_bytes / U->xfer);

                /* 1-based phase number for logs */
                int phase_no_for_log = (int)(idx + gi + 1);
                exec_phase_data_local(U, phase_no_for_log, (uint64_t)Nops, 0);
            }

            double t1 = MPI_Wtime();
            
            /* Per-rank round stats */
            double my_start = 0.0, my_end = 0.0;
            unsigned long long my_ops_ull = 0ULL;

            if (chosen[rank] >= 0) {
                my_start = t0;
                my_end   = t1;
                /* Nops was computed when we executed the phase */
                /* Recompute cheaply here for clarity; keep it in a local when you exec if you prefer */
                {
                    int gi = chosen[rank];
                    phase_t *U = &phases[idx + gi];
                    if (U->xfer > 0) my_ops_ull = (unsigned long long)(U->total_bytes / U->xfer);
                }
            } else {
                /* Sentinels so idle ranks don't disturb MIN/MAX reductions */
                my_start = 1e300;   /* very large */
                my_end   = -1e300;  /* very small */
            }

            /* Round-wide reductions */
            double min_start = 0.0, max_end = 0.0;
            unsigned long long sum_ops_ull = 0ULL;

            MPI_Reduce(&my_start, &min_start,   1, MPI_DOUBLE,               MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&my_end,   &max_end,     1, MPI_DOUBLE,               MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&my_ops_ull, &sum_ops_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                /* Compute phase-number span for *this round* from chosen[] */
                long long pmin = -1, pmax = -1;
                for (int r = 0; r < nprocs; r++) {
                    if (chosen[r] >= 0) {
                        long long p = (long long)idx + (long long)chosen[r] + 1; /* 1-based phase number */
                        if (pmin < 0 || p < pmin) pmin = p;
                        if (pmax < 0 || p > pmax) pmax = p;
                    }
                }

                double wall = (sum_ops_ull > 0) ? (max_end - min_start) : 0.0;

                /* If only one phase ran this round, pmin==pmax, still prints fine */
                fprintf(stdout,
                    "UNIQUE wave SUMMARY: phases [%lld..%lld] ops=%llu wall=%.6fs (min_start=%.6f, max_end=%.6f)\n",
                    (long long)pmin, (long long)pmax,
                    (unsigned long long)sum_ops_ull, wall, min_start, max_end);
                fflush(stdout);
            }
            free(chosen);
        }
        idx += run_len;
        free(done);
    }

    if (rank == 0) {
        fprintf(stderr, "Loaded %zu phases\n", nph);
        fflush(stderr);
    }
    free(phases);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank=0, nprocs=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const char* plan_path=NULL;
    io_api_t ioapi=API_POSIX;
    meta_api_t metapi=META_POSIX;
    col_mode_t cm=COL_NONE;

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--plan") && i+1<argc) {
            plan_path=argv[++i];
        } else if(!strcmp(argv[i],"--io-api") && i+1<argc){
            const char* v=argv[++i];
            if(!strcmp(v,"posix")) ioapi=API_POSIX;
            else if(!strcmp(v,"mpiio")) ioapi=API_MPIIO; /* reserved */
        } else if(!strcmp(argv[i],"--meta-api") && i+1<argc){
            const char* v=argv[++i];
            (void)v; /* POSIX-only for now */
            metapi=META_POSIX;
        } else if(!strcmp(argv[i],"--collective") && i+1<argc){
            const char* v=argv[++i];
            if(!strcmp(v,"none")) cm=COL_NONE;
            else if(!strcmp(v,"independent")) cm=COL_INDEP;
            else if(!strcmp(v,"collective")) cm=COL_COLL;
        }
    }

    if(!plan_path){
        if(rank==0){
            fprintf(stderr,"Usage: %s --plan payload/plan.csv [--io-api posix|mpiio] [--meta-api posix] [--collective none|independent|collective]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    /* All ranks read/iterate plan to stay in step (but only participants open/IO) */
    FILE* fp = fopen(plan_path,"r");
    if(!fp){
        perror("fopen plan");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    run_plan(fp);
    fclose(fp);

    MPI_Barrier(MPI_COMM_WORLD);

    (void)ioapi; (void)metapi; (void)cm; /* reserved */

    MPI_Finalize();
    return 0;
}
