import os


def count_files_in_directory(dir_path):
    """
    Count the number of files in the specified directory and its subdirectories.

    :param dir_path: Path to the directory
    :return: Number of files in the directory
    """
    file_count = 0
    for root, dirs, files in os.walk(dir_path):
        file_count += len(files)
    return file_count
    

def count_lines_in_file(file_path):
    """
    Count the number of lines in the specified file.

    :param file_path: Path to the file
    :return: Number of lines in the file
    """
    line_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            line_count += 1
    return line_count

for dataset, version in [
    ('big2015', 2),
    ('malv2022', 3)
    ]:

    path_files = 'D:\\IME\\gpt-malware\\data-{dataset}\\dataset\\proc-1\\by-label'

    print(dataset)
    print('files')

    n_total = 0
    for i in range(9):
        p = path_files.format(dataset=dataset) + '\\' + str(i)
        n = count_files_in_directory(p)
        print(i+1, n)
        n_total += n
    print('total', n_total)


    print('chunks')

    path_chunks = "D:\\IME\\gpt-malware\\data-{dataset}\\dataset\\proc-1\\version-{version}\\fold-1\\0.limit-all.fold-1.chunk-32.version-{version}.{part}.csv"

    c_total = 0
    for part in ['train', 'test', 'eval']:
        p = path_chunks.format(dataset=dataset, version=version, part=part)
        c = count_lines_in_file(p)
        print(part, c)
        c_total += c
    print('total', c_total)