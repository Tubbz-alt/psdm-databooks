
# Transfer rates for files in and out of HPSS

This data set contains file tranfers to and from HPSS at SLAC for xtc, smalldata xtc and hdf files. For each transfer the file system and the osts (stripes) a files resides on is recorded. Typically a file is not striped and therefore resides only on a single ost.

Irods runs on a single server and all transfers are executed on that server. As we typically allow up to 16 parallel transfers the maximum aggregated transfer rate is limmited to 10Gb/s due to the network connection.

The files are transferred with pftp. For file migrations (writing to HPSS) pftp transfers a file from local disk to the HPSS disk cache. It does not wait for the writing to tape. For file restores (reading from HPSS) the transfer time includes two steps a) transferring a file from tape to the HPSS cache b) transfer from HPSS cache to local disk. 
Therefore restores are usualy slower than migrations.

<dl>
  <dt>instr</dt>
    <dd>instrument</dd>
  <dt>ftype</dt>
    <dd>file type 
        <dl>
        <li> 0: xtc 
        <li> 1: smalldata xtc 
        <li> -1: all other
        </dl>
      </dd> 
  <dt>start</dt>
    <dd>start time of the transfer, unix-time</dd>
  <dt>fsz</dt>
    <dd>file size in bytes</dd>
  <dt>elap</dt>
    <dd>elapsed time in milliseconds for a transfer</dd>
  <dt>ttype</dt>
    <dd>transfer type 
      <dl>
      <li>0: file written to HPSS (migration)
      <li>1: file retrieved from HPSS (restore) 
      <li>-1: unknown
      </dl>
    </dd>
  <dt>fsid</dt>
    <dd>file system id, e.g.: 12 for ana12</dd>
  <dt>nstripe</dt>
    <dd>number of lustre stripes for a file </dd>
  <dt>stripe0</dt>
    <dd>first stripe number</dd>
  <dt>stripeall</dt>
    <dd>csv for all stripes of a file</dd>
</dl>
