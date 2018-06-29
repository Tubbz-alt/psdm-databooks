{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Example querying Prometheus\n",
    "\n",
    "Prometheus is queried using the http api. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time \n",
    "import requests\n",
    "import numpy as np\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "plt.style.use(\"seaborn\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prometheus url"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "promsrv = \"http://localhost:9090\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Query and store the data into two numpy arrays (ib, ost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "# select time interval\n",
    "stop = time.time()\n",
    "start = stop - 60 * 15\n",
    "#stop = 1530226822.0 + 120 \n",
    "#start = stop - 200\n",
    "\n",
    "# Query for psexport infiniband rate \n",
    "query = 'irate(node_infiniband_port_data_received_bytes{group=\"dtn\", instance=~\"psexport08.*\"}[10s])'\n",
    "payload = {'query': query,\n",
    "           'start': int(start), 'end': int(stop), 'step': '5s'}\n",
    "url = \"{}/api/v1/query_range\".format(promsrv)\n",
    "r = requests.get(url, params=payload)\n",
    "\n",
    "data = r.json()\n",
    "point = data['data']['result'][0]\n",
    "ib = np.array(point['values'], float).transpose()\n",
    "ib[1] /= pow(2,20)\n",
    "\n",
    "#Query OST zpool read rate of ana02, sum all osts \n",
    "query_zfs = 'sum(irate(node_zfs_zpool_nread{group=\"ana02\"}[10s]))'\n",
    "payload = {'query': query_zfs,\n",
    "           'start': int(start), 'end': int(stop), 'step': '5s' }\n",
    "\n",
    "url = \"{}/api/v1/query_range\".format(promsrv)\n",
    "r = requests.get(url, params=payload)\n",
    "data = r.json()\n",
    "point = data['data']['result'][0]\n",
    "ost = np.array(point['values'], float).transpose()\n",
    "ost[1] /= pow(2,20)\n",
    "\n",
    "#for point in res: \n",
    "#    #print(point['metric'])\n",
    "#    #print(\"   \", point['values'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Plotting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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XHF27NihTp06KSkpqdTpd955p2JjY53/CgoKdOLECf3tb39TbGysoqKitGfPHknS/v37FRUVpaioKE2aNMk5j3fffVcDBgxQv379tGTJkmLz/+GHHxQWFqb09PQyaywsLNTEiRMVFRWl2NhYHThwoML1ItQBXKLoRawyPvnkE+f1CZ5++mllZGSYWRpQY4cPH9bbb7/t/GKvkgzDUGBgoBYtWuT85+bmpnXr1un+++/XokWL9Mwzz+jVV1+VJL344ouKi4vT8uXLdfLkSX3yySdKS0tTYmKili1bpmXLlikhIUE5OTnO+cfHx6t169bl1rl582adPn1ay5cv14svvqjp06dXuG6EOoBLFL2IVcY777zjvBTlK6+8oqCgIDNLA6pl9uzZ2r59u6QL13l4/fXX5e3tXerYM2fOlPqm9q9//avuvfdeSb9ddCsvL09HjhxRhw4dJP12YagWLVpo6dKlcnd3l4eHhzw9PXX69GlJ0urVq9WpUyc1bdrUOe+NGzcqKipKMTExzr9tP3TokHO+rVq10tGjRyt8s02oAxa3fft2DRw40Hl73LhxWrVqldLT03XvvfcqPj5egwYN0gMPPODcy77++uuVn5+vxx57TO+//74kKTExsdi3wkkXvlhm586dGj16tFJSUnTPPfcoNTVViYmJevrpp/XUU0+pZ8+eev311/XKK68oKipKkZGROnPmjCTpww8/VHR0tAYPHqwRI0YoOzu7jrqCK1nDhg3LfdN65swZnThxQk8++aSioqK0cOFC5zSHw6F+/fppzpw5euqpp5Sdna1GjRo5pxddGMput+t3v/udJOmzzz6Tn5+fgoODlZ2drX//+996+OGHnY/59ddfNWfOHC1cuFCLFy/WsWPHtGvXLl133XX67LPPVFBQoIMHDyotLa3C3xFCHbiCHThwQH379tWSJUt0ww036KOPPio2/R//+Ifmzp2r1NRULViwQJMnTy42PTo6WgEBAZoxY4ZCQ0OLTdu7d6+mT5+uBQsW6I033lDnzp21fPlyeXh4aNu2bTp27Jjmzp2rd955R++++65uv/12zZs3z+xVxhVm8eLFio2N1Zo1azRt2jTFxsZqx44d5T6mYcOGGjlypGbMmKGEhAStWbNGe/fulXQhtFevXq3x48dfchEd6dIL6Xz11VeKj4/XjBkzJEkzZszQyJEj5e7+2xe6pqSk6OjRoxoyZIhiY2OVmpqqo0ePqkuXLrrxxhs1aNAgvfvuu7rmmmsumX9JfE0scAXz8/NT27ZtJUnNmzd3nhsvEhAQoL///e8aMGCApk2bVuxSkxUJDw93XtClsLDQef4yKChIp0+f1pdffimHw6EhQ4ZIkvLy8hQSElJLawZcEBMTo5iYGM2ePVt/+MMfKvVd7N7e3howYIAkycPDQ506ddL333+vM2fO6Prrr5evr6+6dOmisWPHqkmTJsV+b4ouDCVd+ADdc889p7lz5yo4OFiSlJycrB9//FHShTAfPny4pk6dqvDwcCUkJFxSy9NPP+38uXv37sUO2ZeGPXXA4kpe9vH8+fPOn0segixtL8DhcMjX11dHjhyp0nJLzvviPRPDMOTh4aEOHTo4P4i0YsUKvfzyy1VaBmCG77//Xs8++6wMw1B+fr52796ttm3bauPGjVqzZo1zTHBwsBo0aKBrrrlGO3fulPTbhaEKCgoUFxen1157rdib1f/+97/Oi++EhYXp9ddf17XXXqsDBw7oxIkTkqTXXntNGRkZ2r9/v/NowKeffqr27dvLbi8/ttlTByzO29tbGRkZMgxDubm52rNnjzp27Fipxx48eFDr1q3Te++9p0GDBumuu+7SNddcU2yMzWZTbm5uleu68cYbNXHiRDkcDgUEBOijjz5SgwYN1L179yrPC6jIiBEjnD9v2bJFCQkJOnjwoL799lstWrRICxYs0Pz58/X73/9et9xyixo3bqwBAwbIbrfr7rvvVocOHRQSEqJx48bpP//5j/Ly8pyno+Li4vT888+rsLBQN910kzp37qzPPvtM6enpxf7EbcyYMc4Pvl2sYcOGiouL09ChQ+Xh4aH27dsrMDBQAQEBMgxDkZGR8vHxUXx8fIXrSagDFteuXTtdf/316tOnj1q1aqVbbrmlUo8rLCxUXFycJkyYIF9fX40bN07jxo3TsmXLiu2F33nnnRo+fHilXnAuFhQUpAkTJujRRx9Vw4YN5enpWeV5ANXRtWtXde3a9ZL7hw0b5vy5tPPlTZo00fz58y+5PzQ0VEuXLi1235133lnhuftFixY5f+7Zs6d69uxZbLrNZqvyVd64ShsAABbBOXUAACyCUAcAwCIIdQAALIJQBwDAIgh1AAAsglAHAMAiCHUAACyCUAcAwCL+P7qBsyGGvOBgAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(ost[0], ost[1], marker='.', c='r', label=\"ost read\")\n",
    "plt.scatter(ib[0], ib[1], marker='+', c='g', label=\"ib read psexport\")\n",
    "plt.xlabel(\"unix time\")\n",
    "plt.ylabel(\"rate [MiB/s]\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
