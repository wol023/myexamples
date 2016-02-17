#!/bin/bash
sacct > sacct.out
cnt=1
while IFS='' read -r line || [[ -n "$line" ]]; do
        if [ $cnt -gt 2 ]
        then
               tempstr=$line
               tempstr_nowhsp="$(echo -e "${tempstr}" | sed -e 's/^[[:space:]]*//')"
               firstword=${tempstr_nowhsp/%\ */}
               #echo "firstword= $firstword"
               echo "$tempstr"
               [[ "$firstword" = *[^0-9]* ]]
               if [ $? -eq 1 ]
               then
                   scontrol show job $firstword | grep WorkDir
               fi
        else
               let cnt=cnt
        fi
               let cnt=cnt+1
done < "sacct.out"
rm sacct.out

