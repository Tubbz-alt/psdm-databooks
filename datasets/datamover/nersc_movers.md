# SLAC to NERSC data transfers

The LCLS raw data are transferred from SLAC to NERSC. The data transfers nodes (dtn) for each
site (SLAC: psexportNN, NERSC: dtnNN) are used for the transfer. The dtn's at each side are clustered 
using the [XRootD](http://xrootd.org/) framework. Typically 4-6 dtn's are used for each cluster.

Files are transferred using xrdcp with third-party-copy (TPC) mode. In this mode each XRootD cluster
picks a server for the transfer and the one at NERSC pulls the data from SLAC. The node that runs the
transfer tool (xrdcp) is not directly involved in the transfer.

Each transfer is validated by checksumming (md5) the after it has been written to disk on the destination.
The transfer time includes both the transfers itself and reading from disk for checksumming.

# Data

For each file transfer six variables are recorded:

**startt**
> time the transfer started by the data mover. (unix-secs) 

**stopt**
> time the transfer stopped by the data mover. (unix-secs)

**fsize**
> file size in bytes.

**rate**
> transfer rate which is filesize/(stopt - startt). The rate calculations includes
> the start up times (for example starting xrdcp), connection and redirection times to  
> XRootD, the transfer and checksumming. 

**rc**
> return of the transfer. Typically zero (for good transfer) 

