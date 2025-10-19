import os
import subprocess

root_dir = "/mnt/hasanfs/repos/io_synthesizer/analysis"
log_file_path = os.path.join(root_dir, "outputs", "darshan_files_parse_log.txt")

total_count = 0
log_lines = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    darshan_files = [f for f in filenames if f.endswith(".darshan")]
    if darshan_files:
        count = len(darshan_files)
        total_count += count
        log_line = f"{os.path.abspath(dirpath)}: {count}"
        print(log_line)
        log_lines.append(log_line)

        for file in darshan_files:
            file_path = os.path.join(dirpath, file)
            txt_output_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_output_path, "w") as out:
                subprocess.run(["darshan-parser", "--total", "--file", "--perf", "--show-incomplete", file_path], stdout=out, stderr=subprocess.DEVNULL)

summary_line = f"\nTotal .darshan files found: {total_count}"
print(summary_line)
log_lines.append(summary_line)

# Save log to file
with open(log_file_path, "w") as log_file:
    log_file.write("\n".join(log_lines))
