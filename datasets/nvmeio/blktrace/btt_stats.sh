#!/bin/bash
# Use btt to create csv files for different stats. 
# The data were collected with blktrace and parsed with blkparse.
#
# Usage:
#    btt_status.sh <cmd> [options]
#
# Commands:
#    devmap [device-root]
#       print device name and major,minor number. Needs to be run on machine with the device.
#       The default device root is nvme. 
#
#    btt <inputfile> [devmap] 
#       produce btt stats csv files from <inputfile>. Needs to run in directory with
#       the input-file

make_dev_map() {
    # create a device map file. The map is from device name to major,minor number
    # e.g.: nvme0n1 259,0. The device name instead of the numbers will be used. 
    shopt -s nullglob
    for devpath in /sys/block/${1:-nvme}* ; do
        echo -n "$(basename ${devpath}) "
        cat ${devpath}/dev | tr : ,
    done
    echo "looked for /sys/block/${1:-nvme}*" 1>&2
}


make_btt_stats() {
    # temp files are written to tmp/ folder and results to res/
    [[ -e res ]] || mkdir res
    [[ -e tmp ]] || mkdir tmp  

    # create latency and seek stats using btt
    cd tmp
    echo "create stats $(pwd)"
    btt -i ../${1} -M ../$2 \
        -p per_io \
        -m seek_rate -s seeks \
        -l lat -q lat

    echo "make csv files"
    shopt -s nullglob
    for fn in ddd* seeks*dat seek_rate*.dat lat_*dat ; do
        (
            case $fn in
                seeks_*dat) echo "ts,seek,rw" ;;
                seek_rate*dat) echo "ts,srate" ;;
                lat_*dat) echo "ts,lat,rw"
            esac  
            sed -e 's/^ *//' ${fn}  | tr -s ' ' ','
        ) > ../res/${fn%.*}.csv
        echo "Transfor ${fn} to ${fn%.*}.csv" 
    done
    cd -
}


cmd=$1
shift 1
case ${cmd} in
    btt) make_btt_stats ${1:?} ${2:-dev.map} ;;
    devmap) make_dev_map ${1:-nvme} ;;
    *)
         sed -n -e '2,/^[^#]\|^$/ s/^#//p' $0
         exit
esac
