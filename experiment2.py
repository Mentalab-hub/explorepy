import experiment
import bt_connection

bt = bt_connection.Bt_Explore()
bt.connect()


while bt.connected is False:
    bt.reconnect()







print("Some Error happened, shutting down")


