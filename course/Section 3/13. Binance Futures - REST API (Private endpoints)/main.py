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

    binance = BinanceFuturesClient("YOUR_API_KEY_GOES_HERE",
                                   "YOUR_SECRET_KEY_GOES_HERE", True)
    print(binance.get_balances())
    print(binance.place_order("BTCUSDT", "BUY", 0.01, "LIMIT", 20000, "GTC"))
    print(binance.get_order_status("BTCUSDT", 2712672670))
    print(binance.cancel_order("BTCUSDT", 2712672670))

    root = tk.Tk()
    root.mainloop()
