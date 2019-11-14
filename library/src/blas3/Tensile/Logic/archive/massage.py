import os

files = [f for f in os.listdir(".") if (f.startswith('vega20') and f.endswith('DB.yaml')) or \
                                       (f.startswith('vega20') and f.endswith('SB.yaml')) or \
                                       (f.startswith('hip') and f.endswith('.yaml'))]
# print(files)
for f in files:
   with open('ldd_'+f, "w") as fo: 
      with open(f) as fi:  
         with open(f) as fi1:
            line = fi1.readline()
            si = 0
            while ("- [" not in line) or ("- [D" in line):# Copying over first part of original file
               if ("SolutionIndex" in line) and (int(line.split(':')[1]) > si):
                  si = int(line.split(':')[1])
               fo.write(line)
               line = fi1.readline()
            line = fi.readline()
            si = si + 1
            while "- -" not in line:
               line = fi.readline()
            #handle - -
            line = "  -" + line.split('- -')[1]
            while "- [" not in line or ("- [D" in line):#Copying first part of new 
                if "LVPB" in line:
                    fo.write(line+"    LdcEqualsLdd: false\n")
                elif "SolutionIndex" in line:
                     fo.write(line.split(':')[0]+': '+str(int(line.split(':')[1])+si)+'\n')
                elif ("&id" in line) and ("[" in line):
                     fo.write((line.split('[')[0]+'\n').replace('&','*'))
                elif "ReplacementKernel" in line:
                     # print("CHANGING")
                     fo.write(line.split(':')[0]+': false\n')
                elif "LdcEqualsLdd" in line:
                     pass
                else:
                     fo.write(line)
                line = fi.readline()
            fo.write(line)#After copied all text
            line = fi1.readline()
            while "- null" not in line:#Copying second part of original
               fo.write(line)
               line = fi1.readline()
            end = line
            line = fi.readline()
            #handle - - - 
            line = "  -" + line.split('- -')[1]
            while line:#Copying second part of new
                if "    - [" in line:
                     new = line.split(',')
                     new[0] = "    - [" + str(int(new[0].split('[')[1])+si) 
                     new[1] = " " + str(float(new[1].split(']')[0])+0.1) + ']'
                     fo.write(new[0]+','+new[1]+'\n')
                else:
                     fo.write(line)
                line = fi.readline()
for f in files:
   os.remove(f)
   os.rename('ldd_'+f, f)
