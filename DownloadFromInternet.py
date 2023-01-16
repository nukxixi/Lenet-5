import gzip
import os
import urllib
import urllib.request
def download(url):
    u = urllib.request.urlopen(url)
    path = urllib.parse.urlsplit(url)[2]
    filename = os.path.basename(path)
    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)
            print(status, end="")
    return filename

def extract_gz_file(filename):
    with gzip.open(filename, 'rb') as infile:
        with open(filename[0:-3], 'wb') as outfile:
            for line in infile:
                outfile.write(line)

if __name__=='__main__':
    url1 = r"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    url2 = r"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    url3 = r"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    url4 = r"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    dirname = "mnist"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    os.chdir("./"+dirname)
    print("Downloading...")
    file1 = download(url1)
    file2 = download(url2)
    file3 = download(url3)
    file4 = download(url4)
    print("Finished!")
    print("Extracting .gz file...")
    extract_gz_file(file1)
    extract_gz_file(file2)
    extract_gz_file(file3)
    extract_gz_file(file4)
    print("Finished")
    print("MNIST dataset already!")