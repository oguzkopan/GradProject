import tkinter as tk
import logging

from connectors.binance_futures import BinanceFuturesClient


logger = logging.getLogger()

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s :: %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('info.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


if __name__ == '__main__':

    binance = BinanceFuturesClient("CIupMS6t9JcsjSHYw7d3SRM2AJnQ0ZDCYRDShaIegkpPVT89b8eHD0lXlIzqW69v",
                                   "pNfTuC0nWoB3fKCQNzIIbrmPpKqmP4rOzBm2OHrvItp3DFmmLSX8N2QT5YDSatcq", True)

    root = tk.Tk()
    root.mainloop()
