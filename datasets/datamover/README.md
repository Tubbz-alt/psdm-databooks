# LCLS Data Mover Transfer Stats

These notebooks look at the dataset from the LCLS data movers. The data
mover transfer the science data files between different storage resource.
There are currently two datasets: file-transfer-rates and incremental-rates.

## File transfer stats 

<dl>
  <dt>startt</dt>
  <dd>start time of a transfer (unix seconds)</dd>
  <dt>stopt</dt>
  <dd>stop time of a transfer (unix seconds)</dd>
  <dt>fsize</dt>
  <dd>file size in bytes</dd>
  <dt>frate</dt>
  <dd>transfer rate of the whole file in bytes/sec. The rate is reported
  by bbcp and does not include overhead due to starting the transfer tools.
  Therefore the <it>frate</it> might be slightly different from the one
  calculated by fsize/(stopt-startt). This is true in particular for small
  files.</dd>
</dl>   
