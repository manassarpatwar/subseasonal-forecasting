#!/bin/bash
token="bc50a084-bcd9-4ae1-a2c6-abf11d5f2c06" # http://guides.dataverse.org/en/latest/user/account.html#api-token
output="./data/rodeo/"
curl -H "X-Dataverse-key: $token" https://dataverse.harvard.edu/api/datasets/:persistentId/versions/2/files?persistentId=doi:10.7910/DVN/IHBANG | tee >(./download/jq '.data[].dataFile.persistentId' >persistentid) >(./download/jq '.data[].dataFile.filename' >filenames)
paste persistentid filenames > arg
cat arg | xargs -L1 bash -c ' wget -O '$output'$1 -P '$output' https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=$0'
rm filenames persistentid arg
