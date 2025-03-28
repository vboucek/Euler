import os
import pickle
from tqdm import tqdm

# File locations
SRC = None  # Adjust to your actual file path
DST = None  # Directory where chunks will be saved

assert SRC and DST, 'Please download the Pivoting data set, and mark in the code where it is:\nLines 6-7 of /experiments/loaders/split_pivoting.py'


# Parameters
DELTA = 10000  # Chunk duration: 10,000 seconds
DAY = 60**2 * 24  # One day in seconds (for potential extensions)

def split_pivoting():
    base_time = None  # Will hold the first timestamp
    cur_time = 0      # Relative time boundary for chunks

    # Ensure output directory exists
    os.makedirs(os.path.dirname(DST), exist_ok=True)

    # Open the input file and skip the header.
    f_in = open(SRC, 'r')
    header = f_in.readline()

    # Open the first chunk file for writing (no header)
    f_out = open(f'{DST}{cur_time}.txt', 'w+')

    # Dictionary to map original src and dst values to new integer IDs
    nmap = {}
    nid = [0]  # Use list to allow modification inside get_or_add

    def get_or_add(val):
        if val not in nmap:
            nmap[val] = nid[0]
            nid[0] += 1
        return nmap[val]

    # Progress bar for tracking relative time progress
    prog = tqdm(desc='Seconds parsed', total=0)

    line = f_in.readline()
    while line:
        tokens = line.strip().split(',')
        if len(tokens) != 13:
            line = f_in.readline()
            continue  # Skip malformed lines

        try:
            # Convert the timestamp (column 9) from float to int seconds
            ts = int(float(tokens[8]))
        except ValueError:
            line = f_in.readline()
            continue

        # Set base_time on the first valid timestamp
        if base_time is None:
            base_time = ts

        # Calculate the relative timestamp
        rel_ts = ts - base_time

        # Update progress bar (using the difference from the current chunk start)
        prog.update(rel_ts - cur_time)

        # Map src and dst values to integer IDs
        src = tokens[1]
        dst = tokens[2]
        label = tokens[12]
        src_id = get_or_add(src)
        dst_id = get_or_add(dst)

        # Write the processed data with relative timestamp to current chunk file
        f_out.write(f'{rel_ts},{src_id},{dst_id},{label}\n')

        # Rotate file if the relative timestamp exceeds the current chunk boundary
        if rel_ts >= cur_time + DELTA:
            cur_time += DELTA
            f_out.close()
            f_out = open(f'{DST}{cur_time}.txt', 'w+')

        line = f_in.readline()

    f_out.close()
    f_in.close()

    # Save the reverse mapping (ID to original value) as a pickle file
    nmap_rev = [None] * (max(nmap.values()) + 1)
    for key, value in nmap.items():
        nmap_rev[value] = key

    with open(f'{DST}nmap.pkl', 'wb+') as f:
        pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    split_pivoting()
