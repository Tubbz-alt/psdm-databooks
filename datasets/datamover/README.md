# LCLS Data Mover Transfer Stats

These notebooks look at the dataset from the LCLS data movers. The data
mover transfer the science data files between different storage resource.
There are currently two datasets: file-transfer-rates and incremental-rates.

## File transfer stats 

<dl>
  <dt>startt </dt>
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
  <dt>ffbtrans</dt>
    <dd>
    <ul>
    <li>1: file transfer was from dss -> ffb </li>
    <li>0: file transfer was to ana file systems</li>
    </ul>
  </dd>
  <dt>fn</dt>
    <dd>filename (no path)</dd>
  <dt>instr</dt>
    <dd>LCLS instrument the data was collected with</dd>
  <dt>srcquery</dt>
    <dd>query used by data mover to find transfers in database</dd>
  <dt>trgfs</dt>
    <dd>file system the data are written to, e.g.L ffb11, ffb21, ana01, ana12</dd>
  <dt>dmhost</dt>
    <dd>host the data mover was running</dd>
  <dt>localtrans</dt>
    <dd>
        <ul>
        <li>0: only the target file system was mounted on the data mover host</li>
        <li>1: src and trg file systems were locally mounted on the data
               mover host.</li>
        </ul>
    </dd>
  <dt>srchost</dt>
    <dd>host the data file was created</dd>
  <dt>trghost</dt>
    <dd>host the data file is written to</dd>
</dl>   








