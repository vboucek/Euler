import os
import pickle
from tqdm import tqdm

# File locations
SRC = None  # Adapt to your actual file path
DST = None  # Directory where chunks will be saved

assert  SRC and DST, 'Please download the OpTC data set, and mark in the code where it is:\nLines 6-7 of /experiments/loaders/split_optc.py'


# Parameters
DELTA = 10000  # 10,000 seconds chunks
DAY = 60**2 * 24  # Full day in seconds (not critical but useful for extensions)


def split_optc():
    last_time = -1
    cur_time = 0

    os.makedirs(os.path.dirname(DST), exist_ok=True)

    f_in = open(SRC, 'r')
    f_in.readline()  # Read and skip the header line

    f_out = open(f'{DST}{cur_time}.txt', 'w+')  # No header written

    nmap = {}
    nid = [0]

    def get_or_add(ip):
        if ip not in nmap:
            nmap[ip] = nid[0]
            nid[0] += 1
        return nmap[ip]

    prog = tqdm(desc='Seconds parsed', total=0)

    line = f_in.readline()

    while line:
        tokens = line.strip().split(',')

        if len(tokens) != 4:
            line = f_in.readline()
            continue  # Skip malformed lines

        ts, src_ip, dest_ip, label = tokens
        ts = int(ts)

        if last_time == -1:
            last_time = ts

        # Progress bar update
        if ts != last_time:
            prog.update(ts - last_time)
            last_time = ts

        # Map IPs to integers
        src_id = get_or_add(src_ip)
        dest_id = get_or_add(dest_ip)

        # Write line to current chunk file (no header, just data)
        f_out.write(f'{ts},{src_id},{dest_id},{label}\n')

        # If we exceed the time window, rotate file
        if ts >= cur_time + DELTA:
            cur_time += DELTA
            f_out.close()
            f_out = open(f'{DST}{cur_time}.txt', 'w+')  # No header in new file

        line = f_in.readline()

    f_out.close()
    f_in.close()

    # Save IP to ID mapping
    nmap_rev = [None] * (max(nmap.values()) + 1)
    for ip, id_ in nmap.items():
        nmap_rev[id_] = ip

    with open(f'{DST}nmap.pkl', 'wb+') as f:
        pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    split_optc()
