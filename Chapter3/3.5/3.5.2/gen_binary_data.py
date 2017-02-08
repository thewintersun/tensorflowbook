import struct
import codecs

outfile = "file0.bin"

with codecs.open(outfile,'wb') as fw:
    for i in range(10):
        str1 = struct.pack('f',i)
        fw.write(str1)

outfile = "file1.bin"

with codecs.open(outfile,'wb') as fw:
    for i in range(10,20):
        str1 = struct.pack('f',i)
        fw.write(str1)


    
    
