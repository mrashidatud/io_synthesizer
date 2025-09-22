/* =============================================================================
 * MPI Synth I/O Harness (mpi_synthio.c)
 *
 * What this does
 *  - Tiny MPI-IO program that generates I/O operations with precise, per-op
 *    probabilities so Darshan-style OP-LEVEL metrics are matched globally when
 *    combined with IOR output.
 *  - You point it at the SAME files and transfer sizes the synthesizer used for
 *    IOR, and it will:
 *      • alternate read/write via a Markov switch     (--rw-switch-prob)
 *      • choose READ vs WRITE fraction                (--p-write)
 *      • choose sequential vs non-seq per type       (--p-seq-read / --p-seq-write)
 *      • choose consecutive subset per type          (--p-consec-read / --p-consec-write)
 *      • introduce file/memory misalignment          (--p-unaligned-file / --p-unaligned-mem)
 *      • honor layout: shared file or FPP            (--layout shared|fpp)
 *      • perform exactly ceil(-B / -t) ops per rank  (-B bytes per rank, -t transfer size)
 *
 * Key flags (all are 0..1 probabilities unless noted)
 *  -o <path>                  : target file (shared) or per-rank base path (FPP)
 * -t <size>                   : transfer size per operation (e.g., 64k, 1m)
 * -B <bytes>                  : total bytes per rank (ops = ceil(B/t))
 * --layout shared|fpp         : shared file vs file-per-process
 * --p-write <p>               : fraction of ops that are WRITES (else READS)
 * --rw-switch-prob <p>        : probability to flip R↔W after each op
 * --p-seq-read <p>            : for READ ops, fraction that are SEQUENTIAL
 * --p-seq-write <p>           : for WRITE ops, fraction that are SEQUENTIAL
 * --p-consec-read <p>         : for READ ops, fraction that are CONSECUTIVE
 * --p-consec-write <p>        : for WRITE ops, fraction that are CONSECUTIVE
 *                               (the harness enforces CONSECUTIVE ⊂ SEQUENTIAL)
 * --p-unaligned-file <p>      : fraction of ops with misaligned FILE offsets
 * --p-unaligned-mem  <p>      : fraction of ops with misaligned MEMORY buffers
 * --think-us <usec>           : optional: sleep this many microseconds per op
 *
 * Darshan semantics matched
 *  - POSIX_SEQ_READS/WRITES   : next offset > previous end for that op type
 *  - POSIX_CONSEC_READS/WRITES: next offset == previous end for that op type
 *
 * Install location (expected by the synthesizer):
 *  - /mnt/hasanfs/bin/mpi_synthio
 *
 * Build:
 *   mpicc -O3 -Wall -Wextra -o mpi_synthio mpi_synthio.c
 *
 * Example:
 *   mpiexec -n 64 /mnt/hasanfs/bin/mpi_synthio \
 *     -o /fs/bench/ior_shared_sequential_r_small.dat --layout shared \
 *     -t 64k -B 24m \
 *     --p-write 0.20 --rw-switch-prob 0.00 \
 *     --p-seq-read 0.50 --p-seq-write 1.00 \
 *     --p-consec-read 0.50 --p-consec-write 0.00 \
 *     --p-unaligned-file 0.60 --p-unaligned-mem 0.40
 * =============================================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <ctype.h>
#include <limits.h>   // for UINT_MAX
#include <errno.h>

#ifndef MPIIO_ALIGN
#define MPIIO_ALIGN 4096LL
#endif

static void die(int rc, const char* msg) {
  fprintf(stderr, "%s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, rc);
}

static void die_mpi(int rc, const char* where) {
  char err[MPI_MAX_ERROR_STRING]; int len=0;
  MPI_Error_string(rc, err, &len);
  fprintf(stderr, "[rank ?] %s failed: %.*s\n", where, len, err);
  MPI_Abort(MPI_COMM_WORLD, rc);
}

static long long parse_size(const char* s) {
  if (!s || !*s) return -1;
  char *end = NULL;
  double val = strtod(s, &end);
  if (end == s) return -1;
  while (*end && isspace((unsigned char)*end)) ++end;
  long long mul = 1;
  if (*end) {
    if (end[0] == 'k' || end[0] == 'K') mul = 1024LL;
    else if (end[0] == 'm' || end[0] == 'M') mul = 1024LL*1024LL;
    else if (end[0] == 'g' || end[0] == 'G') mul = 1024LL*1024LL*1024LL;
    else if (end[0] == 'b' || end[0] == 'B') mul = 1LL;
    else return -1;
  }
  long long out = (long long)(val * (double)mul);
  return out < 0 ? -1 : out;
}

static inline double urand(unsigned *state){
  // xorshift32; returns uniform in [0,1]
  unsigned x = *state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  *state = x ? x : 1u;
  return (double)x / (double)UINT_MAX;
}

static void usage(int rc) {
  if (rc == 0) {
    fprintf(stdout,
      "mpi_synthio - synthetic MPI-IO harness\n"
      "Required:\n"
      "  -o <path>                target path (shared) or base path (FPP)\n"
      "  -t <size>                transfer size per op (e.g., 64k, 1m)\n"
      "  -B <bytes>               bytes per RANK (ops = ceil(B/t))\n"
      "Options:\n"
      "  --layout shared|fpp      default: shared\n"
      "  --p-write <p>            fraction of ops that are WRITES (default 0.5)\n"
      "  --rw-switch-prob <p>     Markov switch prob R↔W after each op (default 0.0)\n"
      "  --p-seq-read <p>         fraction of READ ops that are sequential\n"
      "  --p-seq-write <p>        fraction of WRITE ops that are sequential\n"
      "  --p-consec-read <p>      fraction of READ ops that are consecutive (⊂ sequential)\n"
      "  --p-consec-write <p>     fraction of WRITE ops that are consecutive (⊂ sequential)\n"
      "  --p-unaligned-file <p>   prob of misaligned FILE offsets (default 0.0)\n"
      "  --p-unaligned-mem  <p>   prob of misaligned MEM buffers (default 0.0)\n"
      "  --think-us <usec>        sleep per op (default 0)\n"
      "\n"
    );
  } else {
    fprintf(stderr, "Try --help\n");
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank=0, nprocs=1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Defaults
  const char* out_path = NULL;
  long long tsize = -1;
  long long B = -1;
  long long think_us = 0;
  long long alignQ = MPIIO_ALIGN;
  int layout_shared = 1; // 1=shared, 0=fpp

  // Probabilities (operation-level)
  double p_write = 0.5;      // fraction of ops that are writes
  double p_rand  = 0.0;      // kept for compatibility; not used for seq/consec logic
  double p_unaligned_file = 0.0;
  double p_unaligned_mem  = 0.0;
  double rw_switch_prob = 0.0;

  // Separate sequential vs consecutive targets per op type (Darshan semantics)
  double p_seq_read = 1.0;      // fraction of READ ops that are sequential (offset > last_read_end)
  double p_seq_write = 1.0;     // fraction of WRITE ops that are sequential
  double p_consec_read = 1.0;   // fraction of READ ops that are consecutive (offset == last_read_end)
  double p_consec_write = 1.0;  // fraction of WRITE ops that are consecutive

  // Parse args
  for (int i=1; i<argc; ++i){
    if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h")) { usage(0); MPI_Finalize(); return 0; }
    else if (!strcmp(argv[i], "-o") && i+1<argc) out_path = argv[++i];
    else if (!strcmp(argv[i], "-t") && i+1<argc) tsize = parse_size(argv[++i]);
    else if (!strcmp(argv[i], "-B") && i+1<argc) B     = parse_size(argv[++i]);
    else if (!strcmp(argv[i], "--layout") && i+1<argc) {
      ++i;
      if (!strcasecmp(argv[i],"shared")) layout_shared = 1;
      else if (!strcasecmp(argv[i],"fpp")) layout_shared = 0;
      else die(2, "invalid --layout (use shared|fpp)");
    }
    else if (!strcmp(argv[i], "--p-write") && i+1<argc) p_write = atof(argv[++i]);
    else if (!strcmp(argv[i], "--p-rand")  && i+1<argc) p_rand  = atof(argv[++i]);
    else if (!strcmp(argv[i], "--p-unaligned-file") && i+1<argc) p_unaligned_file = atof(argv[++i]);
    else if (!strcmp(argv[i], "--p-unaligned-mem")  && i+1<argc) p_unaligned_mem  = atof(argv[++i]);
    else if (!strcmp(argv[i], "--rw-switch-prob")   && i+1<argc) rw_switch_prob   = atof(argv[++i]);

    else if (!strcmp(argv[i], "--p-seq-read")    && i+1<argc) p_seq_read    = atof(argv[++i]);
    else if (!strcmp(argv[i], "--p-seq-write")   && i+1<argc) p_seq_write   = atof(argv[++i]);
    else if (!strcmp(argv[i], "--p-consec-read") && i+1<argc) p_consec_read = atof(argv[++i]);
    else if (!strcmp(argv[i], "--p-consec-write")&& i+1<argc) p_consec_write= atof(argv[++i]);

    else if (!strcmp(argv[i], "--think-us") && i+1<argc) think_us = parse_size(argv[++i]);
    else {
      if (rank==0) fprintf(stderr, "Unknown/invalid arg: %s\n", argv[i]);
      usage(1);
      MPI_Finalize(); return 1;
    }
  }

  if (!out_path || tsize <= 0 || B <= 0) {
    if (rank==0) fprintf(stderr, "Missing required -o / -t / -B\n");
    usage(1);
    MPI_Finalize(); return 1;
  }

  // Clamp probabilities to [0,1]
  if (p_write < 0) p_write = 0; 
  if (p_write > 1) p_write = 1;

  if (p_rand  < 0) p_rand  = 0; 
  if (p_rand  > 1) p_rand  = 1;

  if (p_unaligned_file < 0) p_unaligned_file = 0; 
  if (p_unaligned_file > 1) p_unaligned_file = 1;

  if (p_unaligned_mem  < 0) p_unaligned_mem  = 0; 
  if (p_unaligned_mem  > 1) p_unaligned_mem  = 1;

  if (rw_switch_prob < 0) rw_switch_prob = 0; 
  if (rw_switch_prob > 1) rw_switch_prob = 1;

  if (p_seq_read   < 0) p_seq_read = 0;   
  if (p_seq_read   > 1) p_seq_read = 1;

  if (p_seq_write  < 0) p_seq_write = 0;  
  if (p_seq_write  > 1) p_seq_write = 1;

  if (p_consec_read < 0) p_consec_read = 0; 
  if (p_consec_read > 1) p_consec_read = 1;

  if (p_consec_write < 0) p_consec_write = 0; 
  if (p_consec_write > 1) p_consec_write = 1;

  // Enforce: consecutive ⊂ sequential (per type) via conditional probabilities
  double pc_r_given_seq = 0.0;
  double pc_w_given_seq = 0.0;
  if (p_seq_read  > 0) pc_r_given_seq = p_consec_read  / p_seq_read;
  if (p_seq_write > 0) pc_w_given_seq = p_consec_write / p_seq_write;
  if (pc_r_given_seq > 1.0) pc_r_given_seq = 1.0;
  if (pc_w_given_seq > 1.0) pc_w_given_seq = 1.0;
  if (pc_r_given_seq < 0.0) pc_r_given_seq = 0.0;
  if (pc_w_given_seq < 0.0) pc_w_given_seq = 0.0;

  // Determine operations per rank
  long long nops = (B + tsize - 1) / tsize;  // ceil
  if (nops <= 0) nops = 1;

  // Build per-rank path for FPP
  char path[4096];
  if (!layout_shared) {
    snprintf(path, sizeof(path), "%s.rank%06d", out_path, rank);
  } else {
    snprintf(path, sizeof(path), "%s", out_path);
  }

  // Open file
  MPI_File fh;
  int amode = MPI_MODE_CREATE | MPI_MODE_RDWR;
  int rc = MPI_File_open(MPI_COMM_WORLD, path, amode, MPI_INFO_NULL, &fh);
  // 1) after MPI_File_open(...)
  if (rc != MPI_SUCCESS) {
    int rnk=0; MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
    fprintf(stderr, "[rank %d] MPI_File_open failed for %s\n", rnk, path);
    die_mpi(rc, "MPI_File_open");
  }

  // Allocate buffer (page-align base)
  size_t alloc = (size_t)(tsize + 4096);
  unsigned char *raw = (unsigned char*)malloc(alloc);
  if (!raw) {
    if (rank==0) fprintf(stderr, "malloc failed\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  uintptr_t p = (uintptr_t)raw;
  uintptr_t aligned = (p + 4095) & ~((uintptr_t)4095);
  unsigned char *buf0 = (unsigned char*)aligned;

  // RNG seed
  unsigned seed = 777u ^ (unsigned)(rank * 2654435761u);

  // Track last-end offsets for read/write separately
  long long last_end_r = 0; // previous read end offset
  long long last_end_w = 0; // previous write end offset

  // Initialize starting op type by p_write
  int is_write = (urand(&seed) < p_write) ? 1 : 0;

  for (long long op = 0; op < nops; ++op) {
    // Markov switch for R/W
    if (urand(&seed) < rw_switch_prob) {
      is_write = !is_write;
    } else {
      // otherwise keep current is_write
    }

    // For the chosen op type, decide sequential vs non-seq, and consecutive (subset)
    int do_seq = 0, do_consec = 0;
    if (is_write) {
      do_seq = (urand(&seed) < p_seq_write);
      if (do_seq) {
        do_consec = (urand(&seed) < pc_w_given_seq);
      }
    } else {
      do_seq = (urand(&seed) < p_seq_read);
      if (do_seq) {
        do_consec = (urand(&seed) < pc_r_given_seq);
      }
    }

    // Choose file offset
    long long off = 0;
    if (do_seq) {
      if (is_write) {
        if (do_consec) {
          off = last_end_w;              // consecutive write
        } else {
          off = last_end_w + tsize;      // sequential (gap > 0)
        }
      } else {
        if (do_consec) {
          off = last_end_r;              // consecutive read
        } else {
          off = last_end_r + tsize;      // sequential (gap > 0)
        }
      }
    } else {
      // Non-sequential: choose a random chunk within [0, nops)
      long long idx = (long long)(urand(&seed) * (double)nops);
      if (idx >= nops) idx = nops - 1;
      if (idx < 0) idx = 0;
      off = idx * tsize;
    }

    // Apply file alignment/misalignment
    if (urand(&seed) < p_unaligned_file) {
      long long mis = (alignQ > 1) ? ((op % (alignQ)) ? (op % alignQ) : 1) : 1;
      off += mis; // break alignment
    } else {
      if (alignQ > 0) off = (off / alignQ) * alignQ; // align down
    }

    // Memory misalignment
    unsigned char* buf = buf0;
    if (urand(&seed) < p_unaligned_mem) {
      buf = buf0 + 4; // typical small misalignment
    }

    // Perform IO
    MPI_Offset moff = (MPI_Offset)off;
    int rc2;
    if (is_write) {
      rc2 = MPI_File_write_at(fh, moff, buf, (int)tsize, MPI_BYTE, MPI_STATUS_IGNORE);
      // 2) after MPI_File_write_at(...) (the write branch)
      if (rc2 != MPI_SUCCESS) {
        int rnk=0; MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
        fprintf(stderr, "[rank %d] write_at failed at op=%lld off=%lld t=%lld path=%s\n",
                rnk, op, (long long)off, (long long)tsize, path);
        die_mpi(rc2, "MPI_File_write_at");
      }
      last_end_w = off + tsize;
    } else {
      rc2 = MPI_File_read_at(fh, moff, buf, (int)tsize, MPI_BYTE, MPI_STATUS_IGNORE);
      // 3) after MPI_File_read_at(...) (the read branch)
      if (rc2 != MPI_SUCCESS) {
        int rnk=0; MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
        fprintf(stderr, "[rank %d] read_at failed at op=%lld off=%lld t=%lld path=%s\n",
                rnk, op, (long long)off, (long long)tsize, path);
        die_mpi(rc2, "MPI_File_read_at");
      }
      last_end_r = off + tsize;
    }

    if (think_us > 0) {
      usleep((useconds_t)think_us);
    }
  }

  // 4) after MPI_File_sync(...)
  rc = MPI_File_sync(fh);
  if (rc != MPI_SUCCESS) {
    int rnk=0; MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
    fprintf(stderr, "[rank %d] MPI_File_sync failed for %s\n", rnk, path);
    die_mpi(rc, "MPI_File_sync");
  }
  MPI_File_close(&fh);
  free(raw);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
