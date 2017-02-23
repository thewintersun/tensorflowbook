import struct
import codecs

filename0 = "file0.bin"

with codecs.open(filename0,'wb') as fw:
    for i in range(10):
        str1 = struct.pack('f',i)
        fw.write(str1)

filename1 = "file1.bin"

with codecs.open(filename1,'wb') as fw:
    for i in range(10,20):
        str1 = struct.pack('f',i)
        fw.write(str1)


    
    
