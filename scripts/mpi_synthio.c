/*
 * mpi_synthio.c
 * Single-run, plan-driven harness for synthetic I/O.
 *
 * Key behaviors (matching the planner):
 * - Reads a CSV plan with rows of type:
 *     data,<path>,<total_bytes>,<xfer>,<p_write>,<p_rand>,<p_seq_r>,<p_consec_r>,<p_seq_w>,<p_consec_w>,
 *          <p_ua_file>,<p_ua_mem>,<rw_switch>,<meta_open>,<meta_stat>,<meta_seek>,<meta_sync>,<seed>,<flags>,
 *          <p_rand_fwd_r>,<p_rand_fwd_w>,<p_consec_internal>
 *     meta,<path>,0,1,0,..., meta counters..., seed,meta_only,...
 *
 * - Correct seq/consec semantics:
 *     consecutive ⊂ sequential.
 *     If seq==consec, only issue the 'consec' share and keep rest random (no seq-only).
 *     If seq>consec, add exactly (seq-consec) share as seq-only (stride≥1).
 *
 * - Random-forward: keeps a forward-only pointer to avoid accidental backward seeks.
 *
 * - Alignment:
 *     File-aligned (Lustre) only if both offset and xfer are multiples of 1 MiB.
 *     For phases with p_ua_file<1.0 and xfer%1MiB==0, we snap offset to 1MiB.
 *     Memory alignment: pick buffer = base or base+1 with probability p_ua_mem.
 *
 * - Uniform across files: plan is already split by file (uniform op distribution is enforced by planner).
 *
 * Build:
 *   mpicc -O3 -Wall -Wextra -o mpi_synthio mpi_synthio.c -lm
 *
 * Run:
 *   mpiexec -n 1 ./mpi_synthio --plan /path/plan.csv --io-api posix --meta-api posix --collective none
 */

#define _LARGEFILE64_SOURCE
#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifndef O_LARGEFILE
#define O_LARGEFILE 0
#endif

typedef enum { API_POSIX=0, API_MPIIO=1 } io_api_t;
typedef enum { CMODE_NONE=0, CMODE_INDEP=1, CMODE_COLL=2 } col_mode_t;

typedef struct {
    char   type[16];      // "data" or "meta"
    char   path[1024];
    uint64_t total_bytes;
    uint64_t xfer;

    double p_write;       // 0 for reads phase, 1 for writes phase
    double p_rand;        // prob for random (rest after seq)
    double p_seq_r;       // target sequential (read) fraction
    double p_consec_r;    // target consecutive (read) fraction
    double p_seq_w;       // target sequential (write) fraction
    double p_consec_w;    // target consecutive (write) fraction
    double p_ua_file;     // probability of file unaligned
    double p_ua_mem;      // probability of mem unaligned

    double rw_switch;     // not used (we keep 0)
    int    meta_open;
    int    meta_stat;
    int    meta_seek;
    int    meta_sync;

    unsigned int seed;
    char   flags[64];

    double p_rand_fwd_r;
    double p_rand_fwd_w;
    double p_consec_internal;
} phase_t;

typedef struct {
    io_api_t api;
    col_mode_t collective;
    char plan_path[1024];
} cfg_t;

static inline uint64_t min_u64(uint64_t a, uint64_t b) { return a<b?a:b; }
static inline double clamp01(double x){ if(x<0) return 0; if(x>1) return 1; return x; }

static int parse_args(int argc, char **argv, cfg_t *C){
    memset(C, 0, sizeof(*C));
    C->api = API_POSIX;
    C->collective = CMODE_NONE;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--plan") && i+1<argc){ strncpy(C->plan_path, argv[++i], sizeof(C->plan_path)-1); }
        else if(!strcmp(argv[i],"--io-api") && i+1<argc){
            i++;
            if(!strcmp(argv[i],"posix")) C->api=API_POSIX; else C->api=API_MPIIO;
        }
        else if(!strcmp(argv[i],"--meta-api") && i+1<argc){
            i++; (void)argv[i]; /* currently unused routing */
        }
        else if(!strcmp(argv[i],"--collective") && i+1<argc){
            i++;
            if(!strcmp(argv[i],"none")) C->collective=CMODE_NONE;
            else if(!strcmp(argv[i],"independent")) C->collective=CMODE_INDEP;
            else C->collective=CMODE_COLL;
        }
    }
    if(strlen(C->plan_path)==0){
        fprintf(stderr,"missing --plan <path>\n");
        return -1;
    }
    return 0;
}

static int read_plan(const char *path, phase_t **out, int *n_out){
    FILE *fp = fopen(path, "r");
    if(!fp){ perror("fopen plan"); return -1; }
    char line[4096];
    int cap=32, n=0;
    phase_t *arr = (phase_t*)calloc(cap, sizeof(phase_t));
    if(!arr){ fclose(fp); return -1; }

    while(fgets(line, sizeof(line), fp)){
        // skip header
        if(!strncmp(line, "type,", 5)) continue;
        // remove trailing newline
        char *nl = strchr(line, '\n'); if(nl) *nl = '\0';
        // tolerate empty lines
        if(line[0]=='\0') continue;

        // Tokenize by comma; allow empty tokens
        char *tok[24]; int k=0;
        char *p = line;
        while(p && k<24){
            tok[k++] = p;
            char *c = strchr(p, ',');
            if(c){ *c = '\0'; p = c+1; } else break;
        }
        if(k<2) continue;

        phase_t ph; memset(&ph,0,sizeof(ph));
        strncpy(ph.type, tok[0], sizeof(ph.type)-1);
        strncpy(ph.path, tok[1], sizeof(ph.path)-1);

        // Default sane values
        ph.xfer=1; ph.total_bytes=0;
        ph.p_write=0; ph.p_rand=0; ph.p_seq_r=0; ph.p_consec_r=0; ph.p_seq_w=0; ph.p_consec_w=0;
        ph.p_ua_file=1; ph.p_ua_mem=0; ph.rw_switch=0; ph.meta_open=0; ph.meta_stat=0; ph.meta_seek=0; ph.meta_sync=0;
        ph.seed=1; strncpy(ph.flags,"",sizeof(ph.flags)-1);
        ph.p_rand_fwd_r=0.5; ph.p_rand_fwd_w=0.0; ph.p_consec_internal=0.0;

        // Fill if present
        if(k>2 && tok[2][0]) ph.total_bytes = strtoull(tok[2],NULL,10);
        if(k>3 && tok[3][0]) ph.xfer        = strtoull(tok[3],NULL,10);
        if(k>4 && tok[4][0]) ph.p_write     = atof(tok[4]);
        if(k>5 && tok[5][0]) ph.p_rand      = atof(tok[5]);
        if(k>6 && tok[6][0]) ph.p_seq_r     = atof(tok[6]);
        if(k>7 && tok[7][0]) ph.p_consec_r  = atof(tok[7]);
        if(k>8 && tok[8][0]) ph.p_seq_w     = atof(tok[8]);
        if(k>9 && tok[9][0]) ph.p_consec_w  = atof(tok[9]);
        if(k>10&& tok[10][0]) ph.p_ua_file  = atof(tok[10]);
        if(k>11&& tok[11][0]) ph.p_ua_mem   = atof(tok[11]);
        if(k>12&& tok[12][0]) ph.rw_switch  = atof(tok[12]);
        if(k>13&& tok[13][0]) ph.meta_open  = atoi(tok[13]);
        if(k>14&& tok[14][0]) ph.meta_stat  = atoi(tok[14]);
        if(k>15&& tok[15][0]) ph.meta_seek  = atoi(tok[15]);
        if(k>16&& tok[16][0]) ph.meta_sync  = atoi(tok[16]);
        if(k>17&& tok[17][0]) ph.seed       = (unsigned int)strtoul(tok[17],NULL,10);
        if(k>18&& tok[18][0]) strncpy(ph.flags, tok[18], sizeof(ph.flags)-1);
        if(k>19&& tok[19][0]) ph.p_rand_fwd_r = atof(tok[19]);
        if(k>20&& tok[20][0]) ph.p_rand_fwd_w = atof(tok[20]);
        if(k>21&& tok[21][0]) ph.p_consec_internal = atof(tok[21]);

        // grow
        if(n==cap){ cap*=2; arr=(phase_t*)realloc(arr, cap*sizeof(phase_t)); if(!arr){fclose(fp);return -1;} }
        arr[n++] = ph;
    }
    fclose(fp);
    *out = arr; *n_out = n;
    return 0;
}

static inline uint64_t align_down_1m(uint64_t x){
    const uint64_t ONE_M = 1ULL<<20;
    return (x/ONE_M)*ONE_M;
}

static int run_posix_data(const phase_t *p){
    int fd = open(p->path, (p->p_write>0.5)?(O_CREAT|O_WRONLY|O_LARGEFILE):(O_RDONLY|O_LARGEFILE), 0644);
    if(fd<0){ perror("open"); return -1; }

    // RNG
    srand(p->seed);

    // Pre-allocate buffer (xfer + 1) to allow mem misalignment
    size_t buf_sz = (size_t)p->xfer + 1;
    char *base = (char*)malloc(buf_sz);
    if(!base){ perror("malloc"); close(fd); return -1; }
    memset(base, 0xab, buf_sz);

    uint64_t ops = (p->xfer>0) ? (p->total_bytes / p->xfer) : 0;
    uint64_t prev_end = 0;             // last end offset
    uint64_t rfwd = 0;                 // random-forward baseline
    const uint64_t ONE_M = 1ULL<<20;

    // probabilities (clamped)
    double p_con = clamp01( (p->p_write>0.5)? p->p_consec_w : p->p_consec_r );
    double p_seq = clamp01( (p->p_write>0.5)? p->p_seq_w    : p->p_seq_r );
    if(p_seq < p_con) p_seq = p_con;   // consecutive subset of sequential
    double p_rand = 1.0 - p_seq;
    if(p_rand<0) p_rand=0;

    // We implement: first p_con consecutive; else if (seq>con) do seq-only; else random
    // seq-only uses stride≥1 * xfer so it is sequential but NOT consecutive.
    // random-forward increments rfwd by random strides.

    for(uint64_t i=0;i<ops;i++){
        int do_consec = 0, do_seqonly = 0;
        double r = (double)rand() / (double)RAND_MAX;

        if(r < p_con){
            do_consec = 1;
        } else {
            double r2 = (double)rand() / (double)RAND_MAX;
            double p_seq_only = (p_seq>p_con)? (p_seq - p_con) / (1.0 - p_con) : 0.0;
            if(r2 < p_seq_only) do_seqonly = 1;
        }

        uint64_t off = 0;

        if(do_consec){
            off = prev_end; // adjacent
        } else if(do_seqonly){
            // stride≥1 * xfer so it's sequential but not adjacent
            uint64_t stride_blocks = 1 + (rand()%8); // 1..8
            off = prev_end + stride_blocks * p->xfer;
        } else {
            // random forward
            uint64_t step_blocks = 1 + (rand()%8);
            rfwd += step_blocks * p->xfer;
            off = rfwd;
        }

        // File alignment control:
        // If xfer is multiple of 1MiB, we may generate aligned or unaligned based on p_ua_file.
        int want_unaligned = ((double)rand()/RAND_MAX) < p->p_ua_file ? 1 : 0;
        if(!want_unaligned && (p->xfer % ONE_M == 0)){
            off = align_down_1m(off);
        }

        // Mem alignment:
        char *buf = base;
        int mem_unaligned = ((double)rand()/RAND_MAX) < p->p_ua_mem ? 1 : 0;
        if(mem_unaligned) buf = base + 1;

        ssize_t rc;
        if(p->p_write > 0.5){
            // write
            rc = pwrite(fd, buf, (size_t)p->xfer, (off_t)off);
        } else {
            // read
            rc = pread(fd, buf, (size_t)p->xfer, (off_t)off);
        }
        if(rc < 0){
            perror("pread/pwrite");
            free(base); close(fd);
            return -1;
        }
        prev_end = off + p->xfer;
    }

    free(base);
    close(fd);
    return 0;
}

static int run_posix_meta(const phase_t *p){
    // Issue meta ops against path
    for(int i=0;i<p->meta_open;i++){
        int fd = open(p->path, O_RDONLY|O_LARGEFILE);
        if(fd>=0) close(fd);
    }
    struct stat sb;
    for(int i=0;i<p->meta_stat;i++){ (void)stat(p->path, &sb); }
    // Seek ops simulated via opening and lseek on O_RDONLY
    int fd = open(p->path, O_RDONLY|O_LARGEFILE);
    if(fd>=0){
        for(int i=0;i<p->meta_seek;i++){
            off_t where = (off_t)((i*4096ULL) % (1ULL<<30));
            (void)lseek(fd, where, SEEK_SET);
        }
        // sync family
        for(int i=0;i<p->meta_sync;i++){ (void)fsync(fd); }
        close(fd);
    }
    return 0;
}

/* MPI-IO runner (not used in your current POSIX-only runs, but kept tidy) */
static int run_mpiio_data(const phase_t *p, col_mode_t cm){
    (void)p; (void)cm; /* silence unused warnings, feature kept for future */
    return 0;
}

int main(int argc, char **argv){
    MPI_Init(&argc,&argv);
    int rank=0, nprocs=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    cfg_t C;
    if(parse_args(argc,argv,&C)!=0){
        if(rank==0) fprintf(stderr,"usage: %s --plan <path> [--io-api posix|mpiio] [--collective none|independent|collective]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if(rank==0) fprintf(stderr,"Loading plan: %s\n", C.plan_path);
    phase_t *P = NULL; int nP=0;
    if(read_plan(C.plan_path, &P, &nP)!=0){
        if(rank==0) fprintf(stderr,"Failed reading plan\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if(nP==0){
        if(rank==0) fprintf(stderr,"Empty plan.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for(int i=0;i<nP;i++){
        if(rank==0) fprintf(stderr,"[phase %d] %s %s\n", i, P[i].type, P[i].path);
        if(!strcmp(P[i].type,"data")){
            if(C.api==API_POSIX) { if(run_posix_data(&P[i])!=0){ MPI_Abort(MPI_COMM_WORLD,-1); } }
            else                  { if(run_mpiio_data(&P[i], C.collective)!=0){ MPI_Abort(MPI_COMM_WORLD,-1); } }
        } else if(!strcmp(P[i].type,"meta")){
            if(run_posix_meta(&P[i])!=0){ MPI_Abort(MPI_COMM_WORLD,-1); }
        } else {
            if(rank==0) fprintf(stderr,"[phase %d] Unknown type: %s (skipping)\n", i, P[i].type);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(P);
    MPI_Finalize();
    return 0;
}
