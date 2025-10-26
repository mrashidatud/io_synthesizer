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
 *       * Consec: adjacent; Seq remainder: fixed gap=xfer; Random: descending 32MiB chunks
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
#define CHUNK_RANDOM (32*1024*1024)

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

static void do_posix_data(const phase_t* p, int phase_idx)
{
    int world_rank=0, world_n=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_n);

    if(p->xfer==0 || p->total_bytes==0){ return; }

    /* Global (plan-level) ops for this phase */
    uint64_t Nops = p->total_bytes / p->xfer;
    int is_write = (p->p_write > 0.5);

    /* Shared/unique/legacy scope based on flags (decide BEFORE opening) */
    int is_shared = (strstr(p->flags, "|shared|") != NULL);
    int is_unique = (strstr(p->flags, "|unique|") != NULL);

    int owner = 0;
    if(is_unique){
        owner = (int)(fnv1a_64(p->path) % (unsigned long)world_n);
    }

    /* Decide local ops for this rank */
    uint64_t Nops_local = 0;
    if(is_shared){
        Nops_local = Nops / (uint64_t)world_n;
        if((uint64_t)world_rank < (Nops % (uint64_t)world_n)) Nops_local++;
    } else if(is_unique){
        if(world_rank != owner){ return; } /* non-owners do not open */
        Nops_local = Nops;
    } else {
        if(world_rank != 0){ return; }     /* non-leader do not open */
        Nops_local = Nops;
    }
    if(Nops_local == 0){ return; }

    /* Open only participating ranks now */
    int fd = open(p->path, p->p_write>0.0 ? O_RDWR|O_CREAT : O_RDONLY, 0644);
    if(fd<0){ perror("open"); return; }

    struct stat st;
    if(fstat(fd,&st)!=0){ perror("fstat"); close(fd); return; }
    off_t fsz = st.st_size;

    /* Compute consec/seq/random counts from local ops */
    double p_con, p_seq;
    if(is_write){
        p_con = p->p_consec_w;
        p_seq = p->p_seq_w;
    }else{
        p_con = p->p_consec_r;
        p_seq = p->p_seq_r;
    }
    if(p_con > p_seq) p_con = p_seq;

    uint64_t N_con = (uint64_t)llround((double)Nops_local * p_con);
    uint64_t N_seq = 0;
    if(p_seq > p_con){ N_seq = (uint64_t)llround((double)Nops_local * (p_seq - p_con)); }
    if(N_con + N_seq > Nops_local) {
        if(N_con > Nops_local) N_con=Nops_local;
        N_seq = Nops_local - N_con;
    }
    uint64_t N_rnd = Nops_local - N_con - N_seq;

    /* Large-op file alignment quotas (aligned vs unaligned) */
    int large = (p->xfer >= LUSTRE_FILE_ALIGN) ? 1 : 0;
    uint64_t N_unaligned_total = 0, N_aligned_total = 0;
    if(large){
        N_unaligned_total = (uint64_t)llround(p->p_ua_file * (double)Nops_local);
        if(N_unaligned_total > Nops_local) N_unaligned_total = Nops_local;
        N_aligned_total = Nops_local - N_unaligned_total;
    }
    uint64_t unaligned_left = N_unaligned_total;
    uint64_t aligned_left   = N_aligned_total;

    /* Memory misalignment quota (per-op proportional) */
    uint64_t N_mem_mis = (uint64_t)llround(p->p_ua_mem * (double)Nops_local);
    uint64_t mis_left = N_mem_mis;

    buf_t b = alloc_buffer((size_t)p->xfer);
    if(!b.base){ perror("alloc_buffer"); close(fd); return; }
    void* aligned_ptr = b.base;
    void* mis_ptr = (void*)((uintptr_t)b.base + 1);

    /* ----- Consecutive ----- */
    off_t cur = 0;
    for(uint64_t i=0;i<N_con;i++){
        off_t start = cur;
        if(large){
            if(unaligned_left>0){
                start = force_unaligned_start(start);
                unaligned_left--;
            }else if(aligned_left>0){
                start = force_aligned_start(start);
                aligned_left--;
            }
        }
        start = clamp_range(start, fsz, p->xfer);
        void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
        if(mis_left>0) mis_left--;
        if(do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write)<0){
            perror(is_write?"pwrite":"pread"); break;
        }
        cur = start + (off_t)p->xfer; /* exactly adjacent */
    }

    /* ----- Sequential remainder (gap = xfer) ----- */
    for(uint64_t i=0;i<N_seq;i++){
        off_t start = cur + (off_t)p->xfer; /* seq-but-not-consec with fixed gap */
        if(large){
            if(unaligned_left>0){
                start = force_unaligned_start(start);
                unaligned_left--;
            }else if(aligned_left>0){
                start = force_aligned_start(start);
                aligned_left--;
            }
        }
        start = clamp_range(start, fsz, p->xfer);
        void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
        if(mis_left>0) mis_left--;
        if(do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write)<0){
            perror(is_write?"pwrite":"pread"); break;
        }
        cur = start + (off_t)p->xfer;
    }

    /* ----- Random (descending 32MiB chunks; optional pre-seek-eof) ----- */
    if(N_rnd>0 && p->pre_seek_eof){
        (void)lseek(fd, fsz, SEEK_SET);
    }
    uint64_t nchunks = (fsz>0) ? ( (fsz + CHUNK_RANDOM - 1) / CHUNK_RANDOM ) : 1;
    uint64_t cidx = (nchunks>0? nchunks-1 : 0);
    srand(p->seed + (unsigned int)world_rank * 1337u);

    for(uint64_t i=0;i<N_rnd;i++){
        off_t chunk_start = (off_t)cidx * (off_t)CHUNK_RANDOM;
        off_t chunk_end   = chunk_start + (off_t)CHUNK_RANDOM;
        if(chunk_end > fsz) chunk_end = fsz;
        off_t span = chunk_end - chunk_start;
        if(span < (off_t)p->xfer){
            if(cidx>0) { cidx--; i--; continue; }
            chunk_start = 0; chunk_end = fsz;
            span = chunk_end - chunk_start;
            if(span < (off_t)p->xfer) { break; }
        }

        off_t start = chunk_start;
        if(span > (off_t)p->xfer){
            uint64_t r = (uint64_t)rand();
            off_t delta = (off_t)(r % (uint64_t)(span - (off_t)p->xfer + 1));
            start = chunk_start + delta;
        }

        if(large){
            if(unaligned_left>0){
                start = force_unaligned_start(start);
                unaligned_left--;
            }else if(aligned_left>0){
                /* move to nearest aligned start ≤ current where possible */
                off_t aligned = start & ~(off_t)(LUSTRE_FILE_ALIGN-1);
                if(aligned < chunk_start) aligned = chunk_start;
                if(aligned + (off_t)p->xfer <= chunk_end){
                    start = aligned;
                }else{
                    /* fallback to ceil alignment if floor doesn't fit */
                    start = force_aligned_start(start);
                }
                aligned_left--;
            }
        }

        start = clamp_range(start, fsz, p->xfer);
        void* use_ptr = (mis_left>0) ? mis_ptr : aligned_ptr;
        if(mis_left>0) mis_left--;
        if(do_prw(fd, use_ptr, (size_t)p->xfer, start, is_write)<0){
            perror(is_write?"pwrite":"pread"); break;
        }

        if(cidx>0) cidx--; else cidx = nchunks-1;
    }

    /* Phase summary (stderr) */
    fprintf(stderr,
            "rank %d: phase %d done: path=%s xfer=%" PRIu64 "B ops=%" PRIu64
            " (consec=%" PRIu64 ", seq=%" PRIu64 ", rand=%" PRIu64 ") %s flags=%s\n",
            world_rank, phase_idx, p->path, p->xfer, Nops_local, N_con, N_seq, N_rnd,
            is_write ? "[WRITE]" : "[READ]", p->flags);
    fflush(stderr);

    free(b.base);
    close(fd);
}

static void do_posix_meta(const phase_t* p, int phase_idx)
{
    /* META ops confined here to match planner expectations. */
    for(uint64_t i=0;i<p->meta_open;i++){
        int fd = open(p->path, O_RDONLY|O_CREAT, 0644);
        if(fd>=0) close(fd);
    }
    struct stat st;
    for(uint64_t i=0;i<p->meta_stat;i++){
        (void)stat(p->path, &st);
    }
    int fd = open(p->path, O_RDONLY|O_CREAT, 0644);
    if(fd>=0){
        for(uint64_t i=0;i<p->meta_seek;i++){
            (void)lseek(fd, (off_t)(i%4096), SEEK_SET);
        }
        for(uint64_t i=0;i<p->meta_sync;i++){
            (void)fsync(fd);
        }
        close(fd);
    }

    int world_rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    fprintf(stderr,
            "rank %d: phase %d done [META]: path=%s open=%" PRIu64 " stat=%" PRIu64
            " seek=%" PRIu64 " sync=%" PRIu64 "\n",
            world_rank, phase_idx, p->path, p->meta_open, p->meta_stat, p->meta_seek, p->meta_sync);
    fflush(stderr);
}

static void run_plan(FILE* fp)
{
    char line[MAX_LINE];
    size_t nph=0;
    while(fgets(line,sizeof(line),fp)){
        phase_t p;
        if(parse_line(line,&p)) continue; /* header/empty */

        /* Seed RNG each phase to keep behavior reproducible across ranks (random phase uses rand()) */
        srand(p.seed);

        if(strcmp(p.type,"data")==0){
            do_posix_data(&p, (int)(nph+1));
        } else if(strcmp(p.type,"meta")==0){
            do_posix_meta(&p, (int)(nph+1));
        }
        nph++;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    int world_rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank==0){
        fprintf(stderr,"Loaded %zu phases\n", nph);
        fflush(stderr);
    }
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
