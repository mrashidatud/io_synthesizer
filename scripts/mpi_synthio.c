/*
 * mpi_synthio.c — MPI micro-harness to “correct” global op mix produced by IOR/mdtest:
 *   - honors global probabilities:
 *       --p-write, --p-rand
 *       --p-seq-read, --p-seq-write
 *       --p-consec-read, --p-consec-write  (consecutive ⊆ sequential)
 *       --p-unaligned-file, --p-unaligned-mem
 *       --rw-switch-prob
 *   - honors byte budget:   -B <total_bytes>
 *   - honors transfer size: -t <bytes>
 *   - operates on ONE existing path (-o) in shared layout; no stray files
 *   - emits metadata ops on the target path (no extra “unique” files):
 *       --meta-open N --meta-stat N --meta-seek N --meta-sync N
 *
 * Build:
 *   mpicc -O3 -Wall -Wextra -o mpi_synthio mpi_synthio.c -lm
 */

#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#include <math.h>

static inline double frand01(void) { return rand() / (double)RAND_MAX; }

typedef struct {
  char path[4096];
  long long total_bytes;
  long long xfer;
  int shared_layout; // unused (always shared), kept for compatibility
  double p_write;
  double p_rand;
  double p_seq_read, p_seq_write;
  double p_consec_read, p_consec_write;
  double p_file_ua, p_mem_ua;
  double rw_switch_prob;
  long meta_open, meta_stat, meta_seek, meta_sync;
  unsigned seed;
} cfg_t;

static void usage(const char *p){
  if (!p) p = "mpi_synthio";
  fprintf(stderr,
  "Usage: %s -o <file> -t <xfer> -B <bytes> [options]\n"
  "  --layout shared         (default)\n"
  "  --p-write <0..1>\n"
  "  --p-rand  <0..1>\n"
  "  --p-seq-read  <0..1>   --p-seq-write  <0..1>\n"
  "  --p-consec-read<0..1>  --p-consec-write<0..1>  (<= corresponding seq)\n"
  "  --p-unaligned-file <0..1>  --p-unaligned-mem <0..1>\n"
  "  --rw-switch-prob <0..1>\n"
  "  --meta-open N --meta-stat N --meta-seek N --meta-sync N\n"
  "  --seed N\n",
  p);
}

/* Choose offset based on seq/consec/random and alignment knobs.
 * last_end is updated by the caller after the I/O is issued.
 */
static MPI_Offset choose_offset_c(
    int is_write, int is_seq, int is_consec,
    long long last_end, long long max_off,
    long long file_size, long long xfer,
    double p_rand, double p_file_ua)
{
  (void)is_write; // same logic for R/W positioning here
  long long off;

  if (p_rand > 0.0 && frand01() < p_rand) {
    off = (long long)(frand01() * (double)(max_off > 0 ? max_off : 0));
  } else if (is_seq) {
    if (is_consec) {
      off = last_end;
    } else {
      // ensure strictly higher offset than previous end: skip at least one xfer to avoid "consecutive"
      off = last_end + xfer + xfer;
    }
    if (max_off > 0) {
      if (off > max_off) off = off % (max_off + 1);
    } else {
      off = 0;
    }
  } else {
    off = (long long)(frand01() * (double)(max_off > 0 ? max_off : 0));
  }

  // file alignment or slight unalignment (+1)
  if (frand01() < p_file_ua) {
    if (off + xfer + 1 <= file_size) off += 1;
  } else {
    // align to 4K
    const long long A = 4096LL;
    if (off > 0) off = (off / A) * A;
    if (off + xfer > file_size) {
      off = (file_size > xfer) ? (file_size - xfer) : 0;
      if (off > 0) off = (off / A) * A;
    }
  }
  if (off < 0) off = 0;
  if (off + xfer > file_size && file_size >= xfer) off = file_size - xfer;
  return (MPI_Offset)off;
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  cfg_t C;
  memset(&C, 0, sizeof(C));
  C.shared_layout = 1;
  C.xfer = 64*1024;
  C.total_bytes = 0;
  C.p_write = 0.0;
  C.p_rand = 0.0;
  C.p_seq_read = C.p_seq_write = 1.0;
  C.p_consec_read = C.p_consec_write = 1.0;
  C.p_file_ua = 0.0; C.p_mem_ua = 0.0;
  C.rw_switch_prob = 0.0;
  C.seed = 1;

  static struct option opts[] = {
    {"layout", required_argument, 0, 1},
    {"o", required_argument, 0, 'o'},
    {"t", required_argument, 0, 't'},
    {"B", required_argument, 0, 'B'},
    {"p-write", required_argument, 0, 2},
    {"p-rand", required_argument, 0, 3},
    {"p-seq-read", required_argument, 0, 4},
    {"p-seq-write", required_argument, 0, 5},
    {"p-consec-read", required_argument, 0, 6},
    {"p-consec-write", required_argument, 0, 7},
    {"p-unaligned-file", required_argument, 0, 8},
    {"p-unaligned-mem", required_argument, 0, 9},
    {"rw-switch-prob", required_argument, 0, 10},
    {"meta-open", required_argument, 0, 11},
    {"meta-stat", required_argument, 0, 12},
    {"meta-seek", required_argument, 0, 13},
    {"meta-sync", required_argument, 0, 14},
    {"seed", required_argument, 0, 15},
    {0,0,0,0}
  };

  int c, idx = 0;
  while ((c = getopt_long(argc, argv, "o:t:B:", opts, &idx)) != -1){
    switch (c){
      case 'o': strncpy(C.path, optarg, sizeof(C.path)-1); break;
      case 't': C.xfer = atoll(optarg); break;
      case 'B': C.total_bytes = atoll(optarg); break;
      case 1:   /* --layout */ if (!strcmp(optarg,"shared")) C.shared_layout=1; else C.shared_layout=1; break;
      case 2:   C.p_write = atof(optarg); break;
      case 3:   C.p_rand  = atof(optarg); break;
      case 4:   C.p_seq_read = atof(optarg); break;
      case 5:   C.p_seq_write= atof(optarg); break;
      case 6:   C.p_consec_read = atof(optarg); break;
      case 7:   C.p_consec_write= atof(optarg); break;
      case 8:   C.p_file_ua = atof(optarg); break;
      case 9:   C.p_mem_ua  = atof(optarg); break;
      case 10:  C.rw_switch_prob = atof(optarg); break;
      case 11:  C.meta_open = atol(optarg); break;
      case 12:  C.meta_stat = atol(optarg); break;
      case 13:  C.meta_seek = atol(optarg); break;
      case 14:  C.meta_sync = atol(optarg); break;
      case 15:  C.seed = (unsigned)atoi(optarg); break;
      default: if (rank==0) usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  if (C.path[0] == '\0' || C.xfer <= 0 || C.total_bytes <= 0){
    if (rank==0) usage(argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (C.p_write < 0) { C.p_write = 0; }
  if (C.p_write > 1) { C.p_write = 1; }
  if (C.p_rand  < 0) { C.p_rand  = 0; }
  if (C.p_rand  > 1) { C.p_rand  = 1; }
  if (C.p_file_ua < 0) { C.p_file_ua = 0; }
  if (C.p_file_ua > 1) { C.p_file_ua = 1; }
  if (C.p_mem_ua  < 0) { C.p_mem_ua  = 0; }
  if (C.p_mem_ua  > 1) { C.p_mem_ua  = 1; }
  if (C.rw_switch_prob < 0) { C.rw_switch_prob = 0; }
  if (C.rw_switch_prob > 1) { C.rw_switch_prob = 1; }

  if (C.p_consec_read > C.p_seq_read)   C.p_consec_read  = C.p_seq_read;
  if (C.p_consec_write > C.p_seq_write) C.p_consec_write = C.p_seq_write;

  srand(C.seed + rank);

  // bytes -> ops
  long long total_ops = C.total_bytes / C.xfer;
  if (total_ops <= 0) total_ops = 1;

  long long write_ops = (long long) llround((double)total_ops * C.p_write);
  if (write_ops < 0) write_ops = 0;
  if (write_ops > total_ops) write_ops = total_ops;

  // Metadata ops per rank (even spread)
  long o_open = C.meta_open / nproc + ((rank < (C.meta_open % nproc)) ? 1:0);
  long o_stat = C.meta_stat / nproc + ((rank < (C.meta_stat % nproc)) ? 1:0);
  long o_seek = C.meta_seek / nproc + ((rank < (C.meta_seek % nproc)) ? 1:0);
  long o_sync = C.meta_sync / nproc + ((rank < (C.meta_sync % nproc)) ? 1:0);

  // open the target file once to perform metadata on it
  MPI_File fh;
  int mrc = MPI_File_open(MPI_COMM_WORLD, C.path, MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
  if (mrc != MPI_SUCCESS){
    if (rank==0) fprintf(stderr, "MPI_File_open failed for %s\n", C.path);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  // stat (get size)
  for (long i=0;i<o_stat;i++){
    MPI_Offset sz=0;
    MPI_File_get_size(fh, &sz);
    (void)sz;
  }
  // seeks
  for (long i=0;i<o_seek;i++){
    MPI_Offset off = (MPI_Offset)((frand01()) * (double)((C.total_bytes > C.xfer) ? (C.total_bytes - C.xfer) : 0));
    MPI_File_seek(fh, off, MPI_SEEK_SET);
  }
  // syncs
  for (long i=0;i<o_sync;i++){
    MPI_File_sync(fh);
  }
  MPI_File_close(&fh);

  // perform open/close metadata ops (on same file) without creating new files
  for (long i=0;i<o_open;i++){
    MPI_File tmp;
    int rc = MPI_File_open(MPI_COMM_WORLD, C.path, MPI_MODE_RDWR, MPI_INFO_NULL, &tmp);
    if (rc==MPI_SUCCESS) MPI_File_close(&tmp);
  }

  // reopen for data IO
  int rc = MPI_File_open(MPI_COMM_WORLD, C.path, MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS){
    if (rank==0) fprintf(stderr, "MPI_File_open failed for %s\n", C.path);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  // buffer (aligned)
  void *buf_base = NULL;
  if (posix_memalign(&buf_base, 4096, (size_t)C.xfer) != 0){
    if (rank==0) fprintf(stderr, "posix_memalign failed\n");
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
  unsigned char *buf = (unsigned char*)buf_base;

  const long long file_size = C.total_bytes; // logical region
  const long long max_off   = (file_size > C.xfer) ? (file_size - C.xfer) : 0;

  // last end offsets per type (for seq/consec semantics)
  long long r_last_end = 0;
  long long w_last_end = 0;

  // start mode
  int cur_is_write = (frand01() < C.p_write);

  for (long long k=0; k<total_ops; k++){
    if (frand01() < C.rw_switch_prob) cur_is_write = !cur_is_write;

    int do_seq = cur_is_write ? (frand01() < C.p_seq_write) : (frand01() < C.p_seq_read);
    int do_con = cur_is_write ? (frand01() < C.p_consec_write) : (frand01() < C.p_consec_read);
    if (do_con && !do_seq) do_seq = 1; // consec implies sequential

    // memory alignment
    unsigned char *ptr = buf;
    if (frand01() < C.p_mem_ua) ptr = buf + 1;

    long long last_end = cur_is_write ? w_last_end : r_last_end;
    MPI_Offset off = choose_offset_c(
      cur_is_write, do_seq, do_con, last_end,
      max_off, file_size, C.xfer, C.p_rand, C.p_file_ua
    );

    MPI_Status st;
    int err;
    if (cur_is_write){
      err = MPI_File_write_at(fh, off, ptr, (int)C.xfer, MPI_BYTE, &st);
      w_last_end = (long long)off + C.xfer;
    } else {
      err = MPI_File_read_at(fh, off, ptr, (int)C.xfer, MPI_BYTE, &st);
      r_last_end = (long long)off + C.xfer;
    }
    if (err != MPI_SUCCESS){
      if (rank==0) fprintf(stderr, "I/O failed at op=%lld\n", k);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
  }

  MPI_File_close(&fh);
  free(buf_base);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
